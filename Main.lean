import LeanDL

open DL

/-- 最後の要素を指す、再利用可能な安全な添字。 -/
def lastIndex : Tensor.Index #v[2, 2, 2] :=
  tensor_index% #v[1, 1, 1]

/-- `fill` で作った 2 × 2 × 2 Tensor の各要素に 1 から 8 を書き込む例。 -/
def exampleTensor : Tensor Nat #v[2, 2, 2] :=
  let t := Tensor.fill #v[2, 2, 2] 0
  let t := t.set (tensor_index% #v[0, 0, 0]) 1
  let t := t.set (tensor_index% #v[0, 0, 1]) 2
  let t := t.set (tensor_index% #v[0, 1, 0]) 3
  let t := t.set (tensor_index% #v[0, 1, 1]) 4
  let t := t.set (tensor_index% #v[1, 0, 0]) 5
  let t := t.set (tensor_index% #v[1, 0, 1]) 6
  let t := t.set (tensor_index% #v[1, 1, 0]) 7
  let t := t.set (tensor_index% #v[1, 1, 1]) 8
  t

def main : IO Unit := do
  IO.println exampleTensor
  IO.println s!"exampleTensor[lastIndex] = {exampleTensor[lastIndex]}"
