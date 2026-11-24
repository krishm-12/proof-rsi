"""Final structures. Imported only when the experiment starts final evaluation."""
from curriculum import Problem

FINAL = (
    Problem("nat_parity_roundtrip", "arithmetic", 1, "(n : Nat) : (2 * n + 1) / 2 = n", "omega"),
    Problem("successor_sub", "arithmetic", 1, "(m n : Nat) (h : n ≤ m) : (m + 1) - n = (m - n) + 1", "omega"),
    Problem("product_congruence", "algebra", 1, "(a b c d : Nat) (h : a = b + c) : d * a = d * c + d * b", "rw [h, Nat.add_comm b c, Nat.mul_add]"),
    Problem("multiplicative_cancel", "algebra", 2, "(a b c : Nat) (hc : 0 < c) (h : a * c = b * c) : a = b", "exact Nat.eq_of_mul_eq_mul_right hc h"),
    Problem("min_between", "inequalities", 1, "(a b c : Nat) (h : c ≤ a) (g : c ≤ b) : c ≤ min a b", "exact Nat.le_min.mpr ⟨h, g⟩"),
    Problem("neg_flip", "inequalities", 1, "(a b c : Int) (h : a ≤ b) (g : b < c) : -c < -a", "omega"),
    Problem("resolve_disjunction", "logic", 1, "(P Q : Prop) (h : P ∨ Q) (k : ¬ Q) : P", "cases h with\n| inl hp => exact hp\n| inr hq => exact False.elim (k hq)"),
    Problem("iff_transport", "logic", 2, "(P Q R : Prop) (pq : P ↔ Q) (qr : Q ↔ R) : R ↔ P", "constructor\n· intro hr\n  exact pq.mpr (qr.mpr hr)\n· intro hp\n  exact qr.mp (pq.mp hp)"),
    Problem("induction_parity_steps", "induction", 2, "(n : Nat) (P : Nat → Prop) (h0 : P 0) (h2 : ∀ k, P (2 * k) → P (2 * (k + 1))) : P (2 * n)", "induction n with\n| zero => exact h0\n| succ n ih => exact h2 n ih"),
    Problem("append_accumulator", "induction", 2, "(xs ys : List Nat) (f : List Nat → Nat) (hs : ∀ a t, f (a :: t) = a + f t) (h0 : f [] = 0) : f (xs ++ ys) = f xs + f ys", "induction xs with\n| nil => simp [h0]\n| cons a t ih => simp only [List.cons_append, hs, ih, Nat.add_assoc]"),
)
