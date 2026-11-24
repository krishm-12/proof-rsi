"""Turn a restricted tactic script into a fresh Lean theorem and audit its axioms."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

TACTICS = frozenset("""
  exact apply refine intro intros assumption rfl constructor left right cases case
  induction rcases obtain have specialize simp simpa rw rfl omega ac_rfl
  contradiction exfalso trivial decide subst change show calc
""".split())
FORBIDDEN = frozenset("""
  sorry sorryAx admit axiom constant unsafe partial noncomputable theorem lemma
  def example by do where deriving termination_by decreasing_by run_tac run_elab
  native_decide set_option macro syntax elab open import namespace section end
  initialize mutual opaque attribute universe variable instance trace_state
""".split())
ALLOWED_CHARS = re.compile(r"[a-zA-Z0-9_ \t\n'.,:()\[\]{}=<>+*/@?\-|!≤≥≠¬∧∨↔→←∀∃⟨⟩·⊢⊥∣₀₁₂₃₄₅₆₇₈₉]+")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9']*")
BRANCH = re.compile(r"\|\s+[A-Za-z_][A-Za-z_0-9']*(?:\s+[A-Za-z_][A-Za-z_0-9']*)*\s*=>\s*(.*)")
FENCE = re.compile(r"\A```(?:lean4?|Lean4?)?[^\S\n]*\n([\s\S]+?)\n[^\S\n]*```\s*\Z")
AUDIT = re.compile(r"^(?:'candidate' depends on axioms: \[([^\n]*)\]|'candidate' does not depend on any axioms)$",
                   re.MULTILINE)
CORE_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
HEADER = ("import Lean\nset_option autoImplicit false\nset_option maxHeartbeats 120000\n"
          "set_option maxRecDepth 256\nset_option warningAsError true\n")
# Bump the policy version whenever proof admission or kernel-audit semantics change.
POLICY_VERSION = "restricted-tactics-v2"


def extract(raw):
    raw = textwrap.dedent(raw.strip("\n")).strip()
    fenced = FENCE.fullmatch(raw)
    if fenced:
        raw = textwrap.dedent(fenced[1]).strip("\n")
    if raw == "by":
        raw = ""
    elif raw.startswith("by\n") or raw.startswith("by "):
        raw = textwrap.dedent(raw[3:]).strip("\n")
    if not raw or len(raw) > 3000 or len(raw.splitlines()) > 40:
        raise ValueError("Return a nonempty tactic script without surrounding prose.")
    if not ALLOWED_CHARS.fullmatch(raw) or any(x in raw for x in ("--", "/-", "-/", "\t")):
        raise ValueError("Only proof syntax in the restricted tactic grammar is allowed.")
    if FORBIDDEN.intersection(WORD.findall(raw)):
        raise ValueError("The script contains a disallowed Lean keyword.")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            branch = BRANCH.fullmatch(line)
            if not branch:
                raise ValueError("Use a named cases or induction branch.")
            line = branch[1]
        elif line.startswith("·"):
            line = line[1:].lstrip()
        if not line:
            continue
        # Nested term or tactic quotations and tactic chaining are absent from the grammar.
        head = WORD.match(line)
        if not head or head[0] not in TACTICS:
            raise ValueError("Start each proof line with an allowed tactic or named branch.")
        if "=>" in line and head[0] not in {"case", "exact", "refine", "have", "show"}:
            raise ValueError("Use a separate named branch for this tactic.")
    return raw


def _lean_binary(explicit):
    path = shutil.which(explicit)
    if not path:
        raise RuntimeError(f"Lean executable not found: {explicit}")
    if Path(path).is_symlink() and Path(path).resolve().name == "elan" or ".elan/bin" in path:
        resolved = subprocess.run(["elan", "which", "lean"], capture_output=True, text=True,
                                  timeout=10, check=True)
        path = resolved.stdout.strip()
    return str(Path(path).resolve())


class Verifier:
    def __init__(self, out, lean="lean"):
        self.lean = _lean_binary(lean)
        self.version = subprocess.check_output([self.lean, "--version"], text=True, timeout=10).strip()
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "verification_cache.jsonl"
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record["result"]
        self.calls = 0
        self.hits = 0
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("LEAN_", "ELAN_"))}

    def verify(self, statement, raw, deadline=None, *, fixture=False):
        # Only the static fixture runner can enable fixture mode. Model output always goes through extract.
        try:
            proof = raw if fixture else extract(raw)
        except ValueError as exc:
            return {"ok": False, "kind": "syntax", "diagnostic": str(exc)}
        src = f"{HEADER}theorem candidate {statement} := by\n{textwrap.indent(proof, '  ')}\n#print axioms candidate\n"
        key = hashlib.sha256(f"{self.version}\0{POLICY_VERSION}\0{src}".encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        from time import monotonic
        timeout = min(8, deadline - monotonic()) if deadline is not None else 8
        if timeout <= 0:
            return {"ok": False, "kind": "timeout", "diagnostic": "Verification deadline reached."}
        self.calls += 1
        try:
            with tempfile.TemporaryDirectory(dir=self.out, prefix="lean-") as directory:
                path = Path(directory).resolve() / "candidate.lean"
                path.write_text(src)
                result = subprocess.run([self.lean, "-j", "1", "-M", "2048", str(path)],
                                        cwd=directory, env=self.env, capture_output=True,
                                        text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "kind": "timeout", "diagnostic": "Lean verification timed out."}
        diagnostic = (result.stdout + result.stderr).replace(str(path), "candidate.lean").strip()
        audits = list(AUDIT.finditer(diagnostic))
        if result.returncode or len(audits) != 1:
            record = {"ok": False, "kind": "lean", "diagnostic": diagnostic[-1500:]}
        else:
            axioms = sorted(name.strip() for name in (audits[0][1] or "").split(",") if name.strip())
            if not set(axioms) <= CORE_AXIOMS:
                record = {"ok": False, "kind": "axioms", "diagnostic": "Unauthorized axioms: " + ", ".join(axioms)}
            else:
                record = {"ok": True, "proof": proof, "axioms": axioms}
        self.cache[key] = record
        with self.path.open("a") as file:
            file.write(json.dumps({"key": key, "result": record}, ensure_ascii=False) + "\n")
        return record
