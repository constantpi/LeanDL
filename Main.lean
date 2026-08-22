import LeanDL

open DL

/-- `fill` で作った 2 × 2 × 2 Tensor の各要素に 1 から 8 を書き込む例。 -/
def exampleTensor : Tensor Nat #v[2, 2, 2] :=
  let t := Tensor.fill #v[2, 2, 2] 0
  let t := t.set #v[0, 0, 0] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 1
  let t := t.set #v[0, 0, 1] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 2
  let t := t.set #v[0, 1, 0] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 3
  let t := t.set #v[0, 1, 1] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 4
  let t := t.set #v[1, 0, 0] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 5
  let t := t.set #v[1, 0, 1] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 6
  let t := t.set #v[1, 1, 0] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 7
  t.set #v[1, 1, 1] (by simp [Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get]) 8

def main : IO Unit := do
  IO.println exampleTensor
