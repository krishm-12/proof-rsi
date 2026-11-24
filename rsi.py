"""Run or inspect a local recursive Lean proof experiment."""
import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from curriculum import TRAIN, VALIDATION, Problem, conjecture, training_candidates
from verifier import Verifier

HERE = Path(__file__).resolve().parent
DOMAINS = tuple(dict.fromkeys(problem.domain for problem in TRAIN))
CONJECTURE_PROMPT = (
    'Propose a provable Nat inequality with an explicit additive gap. Output only a single '
    'JSON array of two distinct expressions, left and right, each using both x and y, '
    'nonnegative small integers, +, *, and parentheses. The right side must equal the left '
    'side plus some natural number for all x and y. Choose your own example, distinct from '
    '["x+y", "x+y+2"].'
)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def select_eval(bank, per_domain):
    groups = defaultdict(list)
    for problem in bank:
        groups[problem.domain].append(problem)
    return tuple(group[i] for i in range(per_domain) for group in groups.values() if i < len(group))


def training_order(round_number, levels):
    groups = defaultdict(list)
    for problem in training_candidates(round_number, levels):
        groups[problem.domain].append(problem)
    for group in groups.values():
        group.reverse()
    return [group[i] for i in range(max(map(len, groups.values())))
            for group in groups.values() if i < len(group)]


def wilson(success, count):
    if not count:
        return None
    p, z = success / count, 1.96
    factor = 1 + z*z / count
    middle = (p + z*z / (2*count)) / factor
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / factor
    return [max(0, middle-radius), min(1, middle+radius)]


class Experiment:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.minutes * 60
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.verifier = Verifier(self.out, args.lean)
        self.metrics = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "lean": self.verifier.version, "rounds": [], "loss": [],
        }
        self.rows = {}
        self.status("loading")
        from engine import Engine
        self.engine = Engine(args.model, args.device, args.seed, args.threads)
        self.metrics["device"] = self.engine.device
        self.status("loaded")

    def status(self, stage):
        self.metrics["status"] = stage
        self.flush()
        print(f"[{time.monotonic()-self.started:6.1f}s] {stage}", flush=True)

    def flush(self):
        self.metrics["elapsed_s"] = round(time.monotonic() - self.started, 2)
        self.metrics["lean_subprocesses"] = self.verifier.calls
        self.metrics["verification_cache_hits"] = self.verifier.hits
        if hasattr(self, "engine"):
            self.metrics["tokens"] = self.engine.tokens.copy()
            self.metrics["gpu_peak"] = self.engine.memory()
        atomic_json(self.out / "metrics.json", self.metrics)

    def solve(self, problem, stage, deadline, attempts, *, base=False):
        solved = None
        previous = None
        events = []
        for index in range(attempts):
            if time.monotonic() >= deadline:
                break
            kind = "initial" if index == 0 else "repair" if index == 1 else "resample"
            prompt = self.engine.proof_prompt(problem.statement, previous if index == 1 else None)
            before = self.engine.tokens.copy()
            started = time.monotonic()
            raw, early = self.engine.generate(
                prompt, deadline, self.args.max_tokens, sample=index > 1,
                seed_key=f"{problem.key}:{index}", base=base, statement=problem.statement,
                verifier=self.verifier,
            )
            check = early or self.verifier.verify(problem.statement, raw, deadline)
            # A time-truncated candidate is reported but is excluded from accuracy denominators.
            timed_out = time.monotonic() >= deadline and not check["ok"]
            event = {
                "stage": stage, "key": problem.key, "domain": problem.domain,
                "level": problem.level, "statement": problem.statement, "attempt": kind,
                "raw": raw, "verification": check, "interrupted": timed_out,
                "elapsed_s": round(time.monotonic()-started, 3),
                "prompt_tokens": self.engine.tokens["prompt"] - before["prompt"],
                "generated_tokens": self.engine.tokens["generated"] - before["generated"],
            }
            events.append(event)
            append_json(self.out / "attempts.jsonl", event)
            self.flush()
            if check["ok"]:
                solved = {
                    "key": problem.key, "domain": problem.domain, "level": problem.level,
                    "statement": problem.statement, "proof": check["proof"], "axioms": check["axioms"],
                    "stage": stage, "attempt": kind,
                }
                break
            previous = (raw, check["diagnostic"])
        return solved, events

    @staticmethod
    def summary(pairs, planned):
        attempted = [(row, events) for row, events in pairs if events]
        completed = [(row, events) for row, events in attempted if not events[-1]["interrupted"]]
        successes = sum(row is not None for row, _ in completed)
        kinds = defaultdict(Counter)
        domains = defaultdict(Counter)
        for row, events in completed:
            domains[events[0]["domain"]].update(tested=1, solved=int(row is not None))
        for _, events in attempted:
            for event in events:
                if not event["interrupted"]:
                    kinds[event["attempt"]].update(attempted=1, verified=int(event["verification"]["ok"]))
        by_attempt = dict(kinds)
        for kind in ("initial", "repair", "resample"):
            counts = kinds[kind]
            by_attempt[kind] = {"attempted": counts["attempted"], "verified": counts["verified"],
                                "rate": counts["verified"]/counts["attempted"] if counts["attempted"] else None}
        return {
            "planned": planned, "tested": len(completed), "solved": successes,
            "complete": len(completed) == planned,
            "accuracy": successes/len(completed) if completed else None,
            "wilson_95": wilson(successes, len(completed)),
            "attempts": by_attempt, "domains": dict(domains),
            "solved_keys": [row["key"] for row, _ in completed if row],
        }

    def evaluate(self, bank, stage, deadline, base=False):
        pairs = []
        for item in bank:
            if time.monotonic() >= deadline:
                break
            result = self.solve(item, stage, deadline, self.args.eval_attempts, base=base)
            if result[0]:
                append_json(self.out / "verified_evaluations.jsonl", result[0])
            pairs.append(result)
        return self.summary(pairs, len(bank))

    def collect(self, item, stage, deadline, attempts):
        result = self.solve(item, stage, deadline, attempts)
        row = result[0]
        if row and row["statement"] not in self.rows:
            self.rows[row["statement"]] = row
            append_json(self.out / "verified_train.jsonl", row)
        return result

    def smoke(self):
        self.status("end-to-end smoke")
        e = self.engine
        limit = min(self.deadline, time.monotonic() + 210)
        fixtures = (
            Problem("smoke_reflexive", "arithmetic", 0, "(a : Nat) : a = a"),
            Problem("smoke_and", "logic", 0, "(P Q : Prop) (h : P ∧ Q) : P", "exact h.1"),
        )
        for item in fixtures:
            row, _ = self.solve(item, "smoke", limit, 2, base=True)
            if row:
                break
        if not row:
            raise RuntimeError("Smoke test produced no Lean-verified model proof.")
        pristine = e.state()
        losses = []
        try:
            if e.train([row], limit, 1, self.args.learning_rate, 0, losses.append) != 1:
                raise RuntimeError("Smoke test did not complete a QLoRA step.")
            if not any(value.ne(pristine[key]).any() for key, value in e.state().items()):
                raise RuntimeError("Smoke QLoRA step did not update adapter parameters.")
            path = self.out / "smoke"
            e.model.save_pretrained(path / "adapter")
            append_json(path / "verified.jsonl", row)
            self.metrics["smoke"] = {"proof": row, "loss": losses[0]["loss"]}
        finally:
            e.restore(pristine)
        self.status("smoke verified and trained")

    def run(self):
        args = self.args
        e = self.engine
        if args.smoke_first:
            self.smoke()
        if args.smoke_only:
            return
        bank = select_eval(VALIDATION, args.eval_per_domain)
        self.metrics["validation_keys"] = [item.key for item in bank]
        self.status("baseline validation")
        baseline = self.evaluate(bank, "baseline", self.deadline, base=True)
        self.metrics["baseline"] = baseline
        self.flush()
        if not baseline["complete"]:
            self.status("budget exhausted during baseline")
            return
        validation_seconds = (sum(
            event["elapsed_s"] for event in self.read_attempts()
            if event["stage"] == "baseline"
        ))
        # Reserve the paired final comparison before assigning collection time.
        final_reserve = max(40, 2.8 * validation_seconds)
        round_deadline = self.deadline - final_reserve
        self.metrics["final_reserved_s"] = round(final_reserve, 1)
        best = baseline["solved"]
        champion = None
        levels = dict.fromkeys(DOMAINS, 0)
        for round_number in range(1, args.rounds + 1):
            remaining = round_deadline - time.monotonic()
            if remaining < max(25, 1.1*validation_seconds):
                break
            slice_end = min(round_deadline, time.monotonic() +
                            remaining / (args.rounds - round_number + 1) * 1.18)
            train_end = slice_end - max(15, 1.08*validation_seconds)
            collect_end = min(train_end - 10, time.monotonic() +
                              (train_end - time.monotonic()) * 0.76)
            stage = f"round_{round_number:02d}"
            record = {"round": round_number, "levels": levels.copy(), "accepted": False}
            self.metrics["rounds"].append(record)
            self.status(f"{stage} collection")
            pairs = []
            candidates = [item for item in training_order(round_number, levels)
                          if item.statement not in self.rows][:args.train_problems]
            for item in candidates:
                if time.monotonic() >= collect_end:
                    break
                pairs.append(self.collect(item, stage, collect_end, args.attempts))
            if round_number >= 2 and self.rows and args.conjectures and time.monotonic() + 15 < train_end:
                raw, _ = e.generate(
                    e.chat_prompt([{"role": "user", "content": CONJECTURE_PROMPT}]),
                    min(train_end - 10, time.monotonic() + max(15, validation_seconds/4)),
                    min(args.max_tokens, 40), sample=True, seed_key=f"conjecture:{round_number}",
                )
                item = conjecture(raw, round_number)
                record["conjecture"] = {"raw": raw, "parsed": item is not None, "verified": False}
                if item and item.statement not in self.rows and time.monotonic() + 12 < train_end:
                    result = self.collect(item, stage, train_end - 8, 1)
                    pairs.append(result)
                    record["conjecture"]["verified"] = result[0] is not None
            record["collection"] = self.summary(pairs, len(candidates) +
                                                int(len(pairs) > len(candidates)))
            for domain, counts in record["collection"]["domains"].items():
                rate = counts["solved"] / counts["tested"]
                if rate >= 0.65:
                    levels[domain] = min(2, levels[domain]+1)
                elif counts["tested"] >= 2 and rate < 0.25:
                    levels[domain] = max(0, levels[domain]-1)

            if not any(row for row, _ in pairs) or time.monotonic() >= train_end:
                record["outcome"] = "no new verified data or insufficient training time"
                self.flush()
                continue
            before = e.state()
            new = [row for row in self.rows.values() if row["stage"] == stage]
            replay = [row for row in self.rows.values() if row["stage"] != stage]
            random.Random(args.seed+round_number).shuffle(replay)
            training = new + replay[:len(new)]
            dataset = self.out / f"{stage}_dataset.jsonl"
            for row in training:
                append_json(dataset, row)

            def on_step(step):
                append_json(self.out / "training_loss.jsonl", step)
                self.metrics["loss"].append(step)
                self.flush()

            self.status(f"{stage} training on {len(training)} verified proofs")
            train_start = time.monotonic()
            try:
                steps = e.train(training, train_end, args.epochs, args.learning_rate,
                                round_number, on_step, args.max_steps)
            except RuntimeError as exc:
                e.restore(before)
                record["outcome"] = f"training rolled back: {exc}"
                self.flush()
                continue
            record["training_s"] = round(time.monotonic() - train_start, 2)
            record["steps"] = steps
            if not steps:
                e.restore(before)
                record["outcome"] = "zero optimizer steps"
                self.flush()
                continue
            path = self.out / stage / "adapter"
            e.model.save_pretrained(path)
            record["adapter"] = str(path.relative_to(self.out))
            self.status(f"{stage} validation")
            validation = self.evaluate(bank, stage + "_validation", round_deadline)
            record["validation"] = validation
            if validation["complete"] and validation["solved"] >= best:
                best = validation["solved"]
                champion = record["adapter"]
                record["accepted"] = True
                record["outcome"] = "accepted"
            else:
                e.restore(before)
                record["outcome"] = "rolled back after worse or incomplete validation"
            self.metrics["selected_adapter"] = champion
            atomic_json(path.parent / "selection.json", record)
            self.flush()

        self.metrics["selected_adapter"] = champion
        self.status("final held-out evaluation")
        from final_bank import FINAL

        final = select_eval(FINAL, args.eval_per_domain)
        self.metrics["final_keys"] = [item.key for item in final]
        for item in final:
            if not self.verifier.verify(item.statement, item.reference, self.deadline, fixture=True)["ok"]:
                raise RuntimeError(f"Final benchmark fixture failed: {item.key}")
        if not champion:
            base_result = self.evaluate(final, "final_base", self.deadline, base=True)
            self.metrics["final"] = {"base": base_result, "selected": None}
        else:
            # Interleave the two models per theorem so a time cutoff leaves paired results.
            base_results, selected_results = [], []
            for item in final:
                if time.monotonic() >= self.deadline:
                    break
                for is_base, results in ((True, base_results), (False, selected_results)):
                    stage = "final_base" if is_base else "final_selected"
                    result = self.solve(item, stage, self.deadline, args.eval_attempts, base=is_base)
                    if result[0]:
                        append_json(self.out / "verified_evaluations.jsonl", result[0])
                    results.append(result)
            base_result = self.summary(base_results, len(final))
            selected_result = self.summary(selected_results, len(final))
            self.metrics["final"] = {"base": base_result, "selected": selected_result}
            if base_result["complete"] and selected_result["complete"]:
                base_keys, selected_keys = (set(result["solved_keys"]) for result in
                                            (base_result, selected_result))
                wins, losses = len(selected_keys-base_keys), len(base_keys-selected_keys)
                discordant = wins+losses
                p = min(1, 2*sum(math.comb(discordant, i) for i in range(min(wins, losses)+1))
                        / 2**discordant)
                self.metrics["final"]["paired"] = {"gains": wins, "losses": losses,
                                                    "two_sided_exact_p": p}
        self.metrics["verified_training_count"] = len(self.rows)
        stage = "complete" if all(row is None or row["complete"] for row in
                                  (self.metrics["final"]["base"], self.metrics["final"]["selected"])) else "budget exhausted during final"
        self.status(stage)

    def read_attempts(self):
        path = self.out / "attempts.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                yield json.loads(line)


def plots(folder):
    os.environ.setdefault("MPLCONFIGDIR", str((folder / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads((folder / "metrics.json").read_text())
    scores = []
    if "baseline" in metrics:
        scores.append(("Baseline val", metrics["baseline"]))
    scores.extend((f"Round {r['round']} val" + ("" if r["accepted"] else " (rolled back)"),
                   r["validation"]) for r in metrics["rounds"] if "validation" in r)
    scores.extend((f"Final {name}", result) for name in ("base", "selected")
                  if (result := metrics.get("final", {}).get(name)))
    scores = [(label, row) for label, row in scores if row["tested"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, row) in enumerate(scores):
        ax.bar(i, row["accuracy"], color="#316b91" if row["complete"] else "#95a0aa")
        ax.text(i, row["accuracy"] + 0.025, f"{row['solved']}/{row['tested']}", ha="center")
    ax.set_xticks(range(len(scores)), [label for label, _ in scores], rotation=20, ha="right")
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Lean-verified theorem accuracy")
    fig.tight_layout()
    fig.savefig(folder / "accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = metrics["loss"]
    if steps:
        for round_number in sorted({step["round"] for step in steps}):
            subset = [(i+1, step["loss"]) for i, step in enumerate(steps)
                      if step["round"] == round_number]
            ax.plot(*zip(*subset), marker="o", label=f"Round {round_number}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No optimizer steps recorded", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Answer-token cross-entropy")
    fig.tight_layout()
    fig.savefig(folder / "loss.png", dpi=150)
    plt.close(fig)


def chat(args):
    from engine import Engine
    from peft import load_peft_weights

    adapter = args.adapter
    if args.run:
        metrics = json.loads((args.run / "metrics.json").read_text())
        selected = metrics.get("selected_adapter")
        adapter = args.run / selected if selected else None
    if adapter is None:
        print("No adapter selected. Starting with the original model.")
    e = Engine(args.model, args.device, args.seed, args.threads, trainable=False)
    if adapter is not None:
        e.restore(load_peft_weights(str(adapter), device="cpu"))
    mode = "both" if adapter else "base"
    print("/base, /adapter, /both, /quit. Enter a prompt to stream an answer.")
    while True:
        try:
            prompt = input(f"{mode}> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if prompt in ("/quit", "/exit"):
            return
        if prompt in ("/base", "/adapter", "/both"):
            if prompt != "/base" and adapter is None:
                print("Supply a trained adapter with --adapter or --run.")
            else:
                mode = prompt[1:]
            continue
        if not prompt:
            continue
        rendered = e.chat_prompt([{"role": "user", "content": prompt}])
        for label in (("base", "adapter") if mode == "both" else (mode,)):
            print(f"\n[{label}]", flush=True)
            e.generate(rendered, math.inf, args.max_tokens, base=label == "base", stream=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=HERE / "models/Qwen2.5-Coder-3B-Instruct")
    common.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    common.add_argument("--threads", type=int, default=6)
    common.add_argument("--seed", type=int, default=17)
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--lean", default="lean")
    run.add_argument("--minutes", type=float, default=60)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--learning-rate", type=float, default=0.0002)
    run.add_argument("--max-steps", type=int, default=0)
    run.add_argument("--max-tokens", type=int, default=72)
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument("--train-problems", type=int, default=10)
    run.add_argument("--eval-per-domain", type=int, choices=(1, 2), default=2)
    run.add_argument("--eval-attempts", type=int, default=2)
    run.add_argument("--smoke-first", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--smoke-only", action="store_true")
    run.add_argument("--conjectures", action=argparse.BooleanOptionalAction, default=True)
    cli = sub.add_parser("chat", parents=[common])
    adapter = cli.add_mutually_exclusive_group()
    adapter.add_argument("--adapter", type=Path)
    adapter.add_argument("--run", type=Path)
    cli.add_argument("--max-tokens", type=int, default=192)
    plot = sub.add_parser("plot")
    plot.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 < args.minutes <= 60 or min(args.rounds, args.attempts, args.eval_attempts,
                                               args.epochs, args.train_problems, args.max_tokens) < 1:
            parser.error("Use 0 < minutes <= 60 and positive round, epoch, attempt, and token limits.")
        experiment = Experiment(args)
        try:
            experiment.run()
        except KeyboardInterrupt:
            experiment.status("interrupted")
        except Exception:
            experiment.status("failed")
            raise
        finally:
            plots(experiment.out)
    elif args.command == "chat":
        chat(args)
    else:
        plots(args.folder)


if __name__ == "__main__":
    main()
