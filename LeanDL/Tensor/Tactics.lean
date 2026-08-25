import LeanDL.Tensor.Basic

namespace DL

/-- 連結したshapeの要素数は、それぞれのshapeの要素数の積になる。 -/
@[simp]
theorem shapeSize_append
    {leftRank rightRank : Nat}
    (left : Vector Nat leftRank)
    (right : Vector Nat rightRank) :
    shapeSize (left ++ right) = shapeSize left * shapeSize right := by
  unfold shapeSize
  rw [Vector.foldl_append]
  simpa using
    (Vector.foldl_assoc
      (op := (· * ·))
      (xs := right)
      (a₁ := left.foldl (· * ·) 1)
      (a₂ := 1))

/-- rankの証明によるcastはshapeの要素数を変えない。 -/
@[simp]
theorem shapeSize_cast
    {rank₁ rank₂ : Nat}
    (h : rank₁ = rank₂)
    (shape : Vector Nat rank₁) :
    shapeSize (Vector.cast h shape) = shapeSize shape := by
  subst rank₂
  rfl

/-- 空shapeはscalarを表し、要素数は1になる。 -/
@[simp]
theorem shapeSize_empty :
    shapeSize (#v[] : Vector Nat 0) = 1 := by
  rfl

/-- 1次元shapeの要素数。 -/
@[simp]
theorem shapeSize_singleton (size : Nat) :
    shapeSize #v[size] = size := by
  simp [shapeSize]

/-- 2次元shapeの要素数。 -/
@[simp]
theorem shapeSize_pair (rows cols : Nat) :
    shapeSize #v[rows, cols] = rows * cols := by
  simp [shapeSize]

/-- shape sizeの定型的な展開と簡約を行い、残ったgoalは利用側へ残す。 -/
macro "shape_simp" : tactic =>
  `(tactic|
    simp only [shapeSize_append, shapeSize_cast, shapeSize_empty,
      shapeSize_singleton, shapeSize_pair, Nat.one_mul, Nat.mul_one])

/-- shape sizeの等式を簡約し、残った自然数の等式を `ring` / `omega` で解く。 -/
macro "tensor_shape" : tactic =>
  `(tactic|
    shape_simp <;> try ring <;> try omega)

-- ここから先は検証用の example。

example (batchSize features : Nat) :
    shapeSize (#v[batchSize] ++ #v[features]) = batchSize * features := by
  tensor_shape

example (batchSize rows cols : Nat) :
    shapeSize (#v[batchSize] ++ #v[rows, cols]) =
      batchSize * rows * cols := by
  tensor_shape

example (batchSize features : Nat) :
    shapeSize (#v[batchSize] ++ #v[1, features]) =
      shapeSize (#v[batchSize] ++ #v[features]) := by
  tensor_shape

end DL
