"""Hand-authored, structure-disjoint banks. Reference proofs are verifier fixtures."""
import ast
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    key: str
    domain: str
    level: int
    statement: str
    reference: str = ""


TRAIN = (
    Problem("sum_transport", "arithmetic", 0, "(a b c : Nat) (h : a + b = c + {k}) : a + (b + 2) = c + {k} + 2", "omega"),
    Problem("difference_bound", "arithmetic", 1, "(a b : Nat) (h : b + {k} ≤ a) : {k} ≤ a - b", "omega"),
    Problem("remainder_bound", "arithmetic", 2, "(a : Nat) : a % {k} < {k}", "exact Nat.mod_lt _ (by decide)"),
    Problem("distribute", "algebra", 0, "(a b : Nat) : {k} * (a + b) = {k} * a + {k} * b", "exact Nat.mul_add _ _ _"),
    Problem("factor_difference", "algebra", 1, "(a b : Nat) : a * {k} - b * {k} = (a - b) * {k}", "exact (Nat.sub_mul _ _ _).symm"),
    Problem("nested_distribute", "algebra", 2, "(a b c d : Nat) : (a + b) * (c + d + {k}) = a * c + a * d + a * {k} + (b * c + b * d + b * {k})", "simp only [Nat.add_mul, Nat.mul_add]\nac_rfl"),
    Problem("chain_gap", "inequalities", 0, "(a b c : Nat) (h : a + {k} ≤ b) (g : b < c) : a + {k} < c", "omega"),
    Problem("signed_interval", "inequalities", 1, "(x y : Int) (h : -{k} ≤ x) (g : x + y ≤ 0) : y ≤ {k}", "omega"),
    Problem("product_mono_right", "inequalities", 2, "(a b c : Nat) (h : a + {k} ≤ b) : (a + {k}) * c ≤ b * c", "exact Nat.mul_le_mul_right c h"),
    Problem("branch_join", "logic", 0, "(P Q R : Prop) (h : P ∨ Q) (p : P → R) (q : Q → R) : R", "cases h with\n| inl hp => exact p hp\n| inr hq => exact q hq"),
    Problem("and_curry", "logic", 1, "(P Q R S : Prop) (h : P ∧ Q → R ∧ S) (p : P) (q : Q) : S ∧ R", "have t := h ⟨p, q⟩\nexact ⟨t.2, t.1⟩"),
    Problem("negated_or", "logic", 2, "(P Q : Prop) (h : ¬ (P ∨ Q)) : ¬ P ∧ ¬ Q", "constructor\n· intro hp\n  exact h (Or.inl hp)\n· intro hq\n  exact h (Or.inr hq)"),
    Problem("iter_add", "induction", 0, "(n : Nat) (f : Nat → Nat) (h0 : f 0 = 0) (hs : ∀ i, f (i + 1) = f i + {k}) : f n = n * {k}", "induction n with\n| zero => exact h0\n| succ n ih => rw [hs n, ih, Nat.succ_mul]"),
    Problem("list_preserves", "induction", 1, "(xs : List Nat) (f : List Nat → Nat) (h0 : f [] = {k}) (hs : ∀ a t, f (a :: t) = f t) : f xs = {k}", "induction xs with\n| nil => exact h0\n| cons a t ih => rw [hs, ih]"),
    Problem("iterate_bound", "induction", 2, "(n : Nat) (f : Nat → Nat) (hs : ∀ i, f i + {k} ≤ f (i + 1)) : f 0 ≤ f n", "induction n with\n| zero => exact Nat.le_refl _\n| succ n ih =>\n  have h := hs n\n  omega"),
)

VALIDATION = (
    Problem("subtract_balance", "arithmetic", 0, "(a b c : Nat) (h : b ≤ a) (g : a - b = c) : b + c = a", "omega"),
    Problem("division_reassemble", "arithmetic", 2, "(n : Nat) : n % 5 + 5 * (n / 5) = n", "exact Nat.mod_add_div _ _"),
    Problem("product_swap", "algebra", 0, "(a b c : Nat) : (a * b) * c = (a * c) * b", "simp only [Nat.mul_assoc, Nat.mul_comm b c]"),
    Problem("factor_shared_sum", "algebra", 1, "(a b c : Nat) : a * b + c * b = (a + c) * b", "rw [Nat.add_mul]"),
    Problem("squeeze", "inequalities", 0, "(a b c : Int) (h : a ≤ b) (i : b ≤ c) (j : c ≤ a) : a = c ∧ b = c", "constructor\n· omega\n· omega"),
    Problem("strict_mul", "inequalities", 2, "(a b c : Nat) (h : a < b) (hc : 0 < c) : c * a < c * b", "exact Nat.mul_lt_mul_of_pos_left h hc"),
    Problem("implication_contrapositive", "logic", 0, "(P Q : Prop) (h : P → Q) (nq : ¬ Q) : ¬ P", "intro hp\nexact nq (h hp)"),
    Problem("or_distrib", "logic", 2, "(P Q R : Prop) (h : (P ∧ Q) ∨ (P ∧ R)) : P ∧ (Q ∨ R)", "cases h with\n| inl hpq => exact ⟨hpq.1, Or.inl hpq.2⟩\n| inr hpr => exact ⟨hpr.1, Or.inr hpr.2⟩"),
    Problem("recursive_agreement", "induction", 1, "(n : Nat) (f g : Nat → Nat) (h0 : f 0 = g 0) (hf : ∀ i, f (i + 1) = 2 * f i) (hg : ∀ i, g (i + 1) = 2 * g i) : f n = g n", "induction n with\n| zero => exact h0\n| succ n ih => rw [hf n, hg n, ih]"),
    Problem("list_transfer", "induction", 2, "(xs : List Nat) (P Q : List Nat → Prop) (h0 : Q []) (hs : ∀ a t, Q t → Q (a :: t)) (hp : ∀ t, Q t → P t) : P xs", "apply hp\ninduction xs with\n| nil => exact h0\n| cons a t ih => exact hs a t ih"),
)


def training_candidates(round_number, levels):
    for item in TRAIN:
        if item.level <= levels[item.domain]:
            offset = 2 + (round_number - 1) % 20
            yield Problem(f"{item.key}_{offset}" if "{k}" in item.statement else item.key,
                          item.domain, item.level, item.statement.format(k=offset),
                          item.reference)


def conjecture(raw, number):
    """Decode one bounded expression pair. Only this reserved goal form is built."""
    try:
        pair = json.loads(raw.strip())
        if not isinstance(pair, list) or len(pair) != 2:
            return None

        def expression(source):
            if not isinstance(source, str) or len(source) > 65:
                raise ValueError
            tree = ast.parse(source, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 24:
                raise ValueError

            def emit(node):
                if isinstance(node, ast.Name) and node.id in {"x", "y"}:
                    return node.id
                if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 20:
                    return str(node.value)
                if isinstance(node, ast.BinOp) and type(node.op) in {ast.Add, ast.Mult}:
                    op = "+" if isinstance(node.op, ast.Add) else "*"
                    return f"({emit(node.left)} {op} {emit(node.right)})"
                raise ValueError

            if {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} != {"x", "y"}:
                raise ValueError
            return emit(tree.body)

        lhs, rhs = map(expression, pair)
        if lhs != rhs:
            return Problem(f"conjecture_{number}", "arithmetic", 1,
                           f"(x y : Nat) : ∃ w : Nat, {lhs} + w = {rhs}")
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
