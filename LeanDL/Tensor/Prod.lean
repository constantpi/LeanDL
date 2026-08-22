import Mathlib

namespace DL

theorem List.fold_mul_ne_zero
    (xs : List Nat)
    (init : Nat)
    (h_neq_zero : init ≠ 0)
    (hxs : ∀ x ∈ xs, x ≠ 0) :
    xs.foldl (· * ·) init ≠ 0 := by
  induction xs generalizing init with
  | nil =>
      simpa
  | cons x xs ih =>
      simp only [List.foldl]
      apply ih (init * x)
      · exact Nat.mul_ne_zero h_neq_zero (hxs x (by simp))
      · intro y hy
        apply hxs y
        simp [hy]

theorem Array.foldl_mul_ne_zero
    (xs : Array Nat)
    (hxs : ∀ x ∈ xs, x ≠ 0) :
    xs.foldl (· * ·) 1 ≠ 0 :=
  Array.foldl_rel
    (xs := xs)
    (f := fun acc x => acc * x)
    (g := fun (_ : Unit) (_ : Nat) => ())
    (r := fun acc (_ : Unit) => acc ≠ 0)
    (a := 1)
    (b := ())
    (by simp)
    (by
      intro x hx acc _ hacc
      have ne_zero : x ≠ 0 := hxs x hx
      exact Nat.mul_ne_zero hacc ne_zero)

theorem Vector.foldl_mul_ne_zero
    (rank : Nat)
    (xs : Vector Nat rank)
    (hxs : ∀ i : Fin rank, xs.get i ≠ 0) :
    xs.foldl (· * ·) 1 ≠ 0 :=
  Vector.foldl_rel
    (xs := xs)
    (f := fun acc x => acc * x)
    (g := fun (_ : Unit) (_ : Nat) => ())
    (r := fun acc (_ : Unit) => acc ≠ 0)
    (a := 1)
    (b := ())
    (by simp)
    (by
      intro x hx acc _ hacc
      obtain ⟨i, hi, hget⟩ := Vector.getElem_of_mem hx
      simp [← hget]
      exact ⟨hacc, hxs ⟨i, hi⟩⟩)

private theorem List.foldl_mul_zero (xs : List Nat) :
    xs.foldl (· * ·) 0 = 0 := by
  induction xs with
  | nil =>
      rfl
  | cons x xs ih =>
      simpa [List.foldl] using ih

theorem List.foldl_mul_eq_zero_of_mem
    (xs : List Nat)
    (init : Nat)
    (hzero : 0 ∈ xs) :
    xs.foldl (· * ·) init = 0 := by
  induction xs generalizing init with
  | nil =>
      simp at hzero
  | cons x xs ih =>
      simp only [List.foldl]
      rcases List.mem_cons.mp hzero with hhead | htail
      · subst x
        simpa using List.foldl_mul_zero xs
      · exact ih (init * x) htail

theorem Array.foldl_mul_eq_zero_of_mem
    (xs : Array Nat)
    (init : Nat)
    (hzero : 0 ∈ xs) :
    xs.foldl (· * ·) init = 0 := by
  have hzeroList : 0 ∈ xs.toList := by
    simpa using hzero
  simpa using
    List.foldl_mul_eq_zero_of_mem xs.toList init hzeroList

theorem Vector.foldl_mul_eq_zero_of_mem
    {n : Nat}
    (xs : Vector Nat n)
    (init : Nat)
    (hzero : 0 ∈ xs) :
    xs.foldl (· * ·) init = 0 := by
  have hzeroArray : 0 ∈ xs.toArray := by
    simpa using hzero
  change xs.toArray.foldl (· * ·) init = 0
  exact
    Array.foldl_mul_eq_zero_of_mem xs.toArray init hzeroArray

end DL
