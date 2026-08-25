import LeanDL.Tensor.Shape

namespace DL.Tensor.Internal

/-- broadcast後の添字を、broadcast前のshapeに対応する添字へ変換する。 -/
def broadcastIndex {rankResult rankSource : Nat}
    (resultShape : Vector Nat rankResult)
    (sourceShape : Vector Nat rankSource)
    (hRank : rankSource ≤ rankResult)
    (hNonzero : ¬shape_is_zero sourceShape)
    (index : Index resultShape) : Index sourceShape :=
  let sourceIndexValues : Vector Nat rankSource := Vector.ofFn fun i =>
    let sourceDim := sourceShape.get i
    let i' : Fin rankResult := ⟨i.val + (rankResult - rankSource), by omega⟩
    let index_i := index.values.get i'
    if index_i < sourceDim then index_i else 0

  have isValid : index_in_bounds sourceShape sourceIndexValues := by
    simp [index_in_bounds, sourceIndexValues]
    intro i
    let i' : Fin rankResult := ⟨i.val + (rankResult - rankSource), by omega⟩
    split
    · rename_i hIndex
      exact hIndex
    · rw [shape_nonzero_iff] at hNonzero
      have := hNonzero i
      omega

  {
    values := sourceIndexValues
    isValid := isValid
  }

/--
削除済みの軸を添字へ戻す。

`axisIndex` が削除した軸の値で、それ以外の値は `index` から引き継ぐ。
-/
def insertAxisIndex
    {rank : Nat}
    {shape : Vector Nat rank}
    (axis : Fin rank)
    (index : Index (eraseAxis shape axis))
    (axisIndex : Fin (shape.get axis)) : Index shape :=
  let values : Vector Nat rank := Vector.ofFn fun i =>
    if hBefore : i.val < axis.val then
      index.values.get ⟨i.val, by omega⟩
    else if hAxis : i.val = axis.val then
      axisIndex.val
    else
      index.values.get ⟨i.val - 1, by omega⟩
  have isValid : index_in_bounds shape values := by
    intro i
    change values[i.val] < shape[i.val]
    simp only [values, Vector.getElem_ofFn]
    split
    · rename_i hBefore
      have h := index.isValid ⟨i.val, by omega⟩
      change index.values[i.val] <
        (shape.eraseIdx axis.val axis.isLt)[i.val] at h
      rw [Vector.getElem_eraseIdx_of_lt axis.isLt (by omega) hBefore] at h
      simpa [Vector.get] using h
    · rename_i hNotBefore
      split
      · rename_i hAxis
        have hi : i = axis := Fin.ext hAxis
        subst i
        exact axisIndex.isLt
      · rename_i hNotAxis
        have hAfter : axis.val < i.val := by omega
        have h := index.isValid ⟨i.val - 1, by omega⟩
        have hIndex : i.val - 1 + 1 = i.val := by omega
        change index.values[i.val - 1] <
          (shape.eraseIdx axis.val axis.isLt)[i.val - 1] at h
        rw [Vector.getElem_eraseIdx_of_ge axis.isLt (by omega) (by omega)] at h
        simpa [Vector.get, hIndex] using h
  { values, isValid }

/-- 2つの安全な添字を、連結したshapeの安全な添字へ結合する。 -/
def appendIndex
    {leftRank rightRank : Nat}
    {leftShape : Vector Nat leftRank}
    {rightShape : Vector Nat rightRank}
    (leftIndex : Index leftShape)
    (rightIndex : Index rightShape) :
    Index (leftShape ++ rightShape) where
  values := leftIndex.values ++ rightIndex.values
  isValid := by
    intro i
    change
      (leftIndex.values ++ rightIndex.values)[i.val] <
        (leftShape ++ rightShape)[i.val]
    rw [Vector.getElem_append, Vector.getElem_append]
    split
    · exact leftIndex.isValid ⟨i.val, by omega⟩
    · exact rightIndex.isValid ⟨i.val - leftRank, by omega⟩

/-- 連結shapeの安全な添字から、先頭側の添字を取り出す。 -/
def prefixIndex
    {leftRank rightRank : Nat}
    {leftShape : Vector Nat leftRank}
    {rightShape : Vector Nat rightRank}
    (index : Index (leftShape ++ rightShape)) : Index leftShape where
  values := Vector.ofFn fun i => index.values.get ⟨i.val, by omega⟩
  isValid := by
    intro i
    have h := index.isValid ⟨i.val, by omega⟩
    simpa [Vector.get, Vector.getElem_append] using h

end DL.Tensor.Internal
