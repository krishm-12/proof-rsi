"""CPU-only checks for the verifier, curriculum, and published evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from curriculum import TRAIN, VALIDATION, conjecture, training_candidates
from final_bank import FINAL
from verifier import Verifier, extract

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class LeanTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.verifier = Verifier(directory.name)

    def test_examples_and_recorded_model_proof(self):
        rows = jsonl(ROOT / "examples/fixtures.jsonl")
        rows += jsonl(ROOT / "results/gpu-rsi-resample/verified_smoke.jsonl")
        for row in rows:
            with self.subTest(key=row["key"]):
                result = self.verifier.verify(row["statement"], row["proof"])
                self.assertTrue(result["ok"], result.get("diagnostic"))
                if "axioms" in row:
                    self.assertEqual(result["axioms"], row["axioms"])

    def test_open_goal_fails_and_reuses_checked_result(self):
        statement = "(P Q : Prop) (h : P) : Q"
        result = self.verifier.verify(statement, "exact h")
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "lean")
        self.assertEqual(self.verifier.verify(statement, "exact h"), result)
        self.assertEqual(self.verifier.calls, 1)
        self.assertEqual(self.verifier.hits, 1)

    def test_rejects_untrusted_commands(self):
        for proof in ("sorry", "run_tac trivial", "exact rfl\naxiom injected : False"):
            with self.subTest(proof=proof), self.assertRaises(ValueError):
                extract(proof)

    def test_evaluation_reference_proofs(self):
        for problem in (*VALIDATION, *FINAL):
            with self.subTest(key=problem.key):
                result = self.verifier.verify(problem.statement, problem.reference, fixture=True)
                self.assertTrue(result["ok"], result.get("diagnostic"))


class ArtifactTests(unittest.TestCase):
    def test_curriculum_and_eval_banks(self):
        domains = {problem.domain for problem in TRAIN}
        self.assertEqual(len(domains), 5)
        keys = [{problem.key for problem in bank} for bank in (TRAIN, VALIDATION, FINAL)]
        self.assertEqual(len(set.union(*keys)), sum(map(len, keys)))
        self.assertEqual(len(VALIDATION), 10)
        self.assertEqual(len(FINAL), 10)
        rows = list(training_candidates(1, dict.fromkeys(domains, 0)))
        self.assertEqual({row.domain for row in rows}, domains)
        self.assertTrue(all(row.level == 0 and "{k}" not in row.statement for row in rows))
        self.assertIsNotNone(conjecture('["x+y","x+y+3"]', 2))
        self.assertIsNone(conjecture('["x+y","x**y"]', 2))

    def test_evidence_totals_and_missing_adapter_results(self):
        attempts = generated = 0
        for folder in sorted((ROOT / "results").glob("gpu-rsi*")):
            metrics = json.loads((folder / "metrics.json").read_text())
            evidence = metrics["evidence"]
            self.assertEqual(len(jsonl(folder / "attempts.jsonl")), evidence["sample_size"])
            self.assertEqual(sum(sum(row.values()) for row in evidence["attempt_outcomes"].values()),
                             evidence["attempts_total"])
            self.assertIsNone(metrics["final"]["selected"])
            self.assertEqual(metrics["verified_training_count"], 0)
            attempts += evidence["attempts_total"]
            generated += metrics["tokens"]["generated"]
        self.assertEqual((attempts, generated), (187, 6841))


if __name__ == "__main__":
    unittest.main()
