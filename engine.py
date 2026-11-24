"""The local Qwen model, sampled proof continuations, and answer-only QLoRA."""
import hashlib
import os
import random
import textwrap
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, TextStreamer)
from transformers.utils import logging

from verifier import extract

SYSTEM = (
    "Lean 4 core (import Lean, no mathlib). Output tactics only, one per line. "
    "Parameters and hypotheses in the theorem header are already in scope. "
    "For (P : Prop) (hp : P) : P, write exact hp. "
    "Use omega for linear Nat and Int arithmetic. Standard names include Nat.add_comm. "
    "Use named induction branches and indent nested tactics."
)


class Deadline(StoppingCriteria):
    def __init__(self, stop_at):
        self.stop_at = stop_at

    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= self.stop_at


class StopAtProof(Deadline):
    def __init__(self, stop_at, tokenizer, statement, prompt_length, verifier):
        super().__init__(stop_at)
        self.tokenizer = tokenizer
        self.statement = statement
        self.prompt_length = prompt_length
        self.verifier = verifier
        self.checked = set()
        self.result = None

    def __call__(self, input_ids, scores, **kwargs):
        if super().__call__(input_ids, scores):
            return True
        raw = self.tokenizer.decode(input_ids[0, self.prompt_length:].tolist(),
                                    skip_special_tokens=True)
        # Try completed tactic lines only. Lean decides whether a prefix closes all goals.
        prefix = raw.rpartition("\n")[0].rstrip()
        if prefix and prefix not in self.checked:
            self.checked.add(prefix)
            try:
                proof = extract(prefix)
            except ValueError:
                return False
            result = self.verifier.verify(self.statement, proof, self.stop_at)
            if result["ok"]:
                self.result = result
                return True
        return False


class Engine:
    def __init__(self, model_path, device, seed, threads, *, trainable=True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        logging.disable_progress_bar()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it in this session.")
        self.device = device
        self.dtype = torch.bfloat16
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            self.dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, device_map={"": device}, dtype=self.dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            ),
        )
        self.model = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "o_proj"],
        ))
        if trainable and device == "cuda":
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()
        self.tokens = {"prompt": 0, "generated": 0, "training_input": 0, "training_answer": 0}
        self.seed = seed

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def restore(self, state):
        set_peft_model_state_dict(self.model, state)

    def memory(self):
        if self.device != "cuda":
            return None
        return {
            "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }

    def chat_prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def proof_prompt(self, statement, previous=None):
        user = f"Complete this Lean 4 proof body:\ntheorem candidate {statement} := by"
        if previous:
            raw, diagnostic = previous
            user += (f"\nEarlier attempt:\n{raw[:650]}\n"
                     f"Verifier feedback:\n{diagnostic[:850]}\nFix the reported error. Return the complete corrected tactic script.")
        return self.chat_prompt([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": user}]) + "by\n"

    def generate(self, prompt, deadline, max_tokens, *, sample=False, seed_key="", base=False,
                 statement=None, verifier=None, stream=False):
        if time.monotonic() >= deadline:
            return "", None
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        size = inputs.input_ids.shape[1]
        stopping = (StopAtProof(deadline, self.tokenizer, statement, size, verifier)
                    if verifier else Deadline(deadline))
        if sample:
            tag = hashlib.sha256(seed_key.encode()).digest()
            torch.manual_seed((self.seed + int.from_bytes(tag[:4], "little")) % 2**32)
        kwargs = {"temperature": 0.65, "top_p": 0.9} if sample else {}
        if stream:
            kwargs["streamer"] = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode(), self.model.disable_adapter() if base else nullcontext():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=sample, stopping_criteria=[stopping],
                pad_token_id=self.tokenizer.eos_token_id, use_cache=True, **kwargs,
            )
        ids = output[0, size:]
        self.tokens["prompt"] += size
        self.tokens["generated"] += ids.numel()
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True), getattr(stopping, "result", None)

    def train(self, rows, deadline, epochs, learning_rate, round_number, on_step, max_steps=0):
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                      lr=learning_rate, weight_decay=0)
        generator = random.Random(self.seed + round_number)
        scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        self.model.train()
        steps = 0
        try:
            for epoch in range(epochs):
                order = list(rows)
                generator.shuffle(order)
                for row in order:
                    if time.monotonic() >= deadline or (max_steps and steps >= max_steps):
                        return steps
                    prompt = self.proof_prompt(row["statement"])
                    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    answer = textwrap.indent(row["proof"], "  ") + "\n"
                    answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
                    answer_ids.append(self.tokenizer.eos_token_id)
                    sequence = prompt_ids + answer_ids
                    if len(sequence) > 768:
                        continue
                    input_ids = torch.tensor([sequence], device=self.device)
                    # Keep the last prompt prediction and the answer predictions. Earlier
                    # prompt positions have zero loss, while prompt context stays visible.
                    labels = torch.full_like(input_ids, -100)
                    labels[0, len(prompt_ids):] = input_ids[0, len(prompt_ids):]
                    target = labels[:, -len(answer_ids):]
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(input_ids=input_ids, use_cache=False,
                                        logits_to_keep=len(answer_ids) + 1)
                    logits = output.logits[:, :-1, :].float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.flatten())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad), 1.0,
                        error_if_nonfinite=not scaler.is_enabled(),
                    )
                    if not torch.isfinite(norm):
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    steps += 1
                    self.tokens["training_input"] += len(sequence)
                    self.tokens["training_answer"] += len(answer_ids)
                    on_step({"round": round_number, "epoch": epoch + 1,
                             "step": steps, "key": row["key"], "loss": float(loss.detach())})
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.eval()
        return steps
