import LeanDL.Tensor.Basic

namespace DL.Tensor

/--
shape の総要素数が等しいとき、平坦なデータの並びを変えずに shape を変更する。
-/
def reshape
    {α : Type}
    {oldRank newRank : Nat}
    {oldShape : Vector Nat oldRank}
    (tensor : Tensor α oldShape)
    (newShape : Vector Nat newRank)
    (hSize : shapeSize oldShape = shapeSize newShape) :
    Tensor α newShape where
  data := tensor.data
  hsize := tensor.hsize.trans hSize

/--
`axes` が軸の置換であることを表す。

各要素は `Fin rank` なので範囲内であることは型が保証する。さらに `Nodup` により、
すべての軸が重複なくちょうど1回ずつ現れることを保証する。
-/
abbrev IsAxisPermutation {rank : Nat}
    (axes : Vector (Fin rank) rank) : Prop :=
  axes.toList.Nodup

/-- `axes[i]` を入力軸として、transpose 後の shape を計算する。 -/
def transposeShape
    {rank : Nat}
    (shape : Vector Nat rank)
    (axes : Vector (Fin rank) rank) : Vector Nat rank :=
  Vector.ofFn fun i => shape.get (axes.get i)

/-- axesからi番目を返す関数 -/
private def axesFunction
    {rank : Nat}
    (axes : Vector (Fin rank) rank) : Fin rank → Fin rank :=
  axes.get

/-- axesFunction が単射であることを示す定理 -/
private theorem axesFunction_injective
    {rank : Nat}
    (axes : Vector (Fin rank) rank)
    (hPermutation : IsAxisPermutation axes) :
    Function.Injective (axesFunction axes) := by
  intro i j hij
  have hList := List.nodup_iff_injective_getElem.mp hPermutation
  let i' : Fin axes.toList.length := ⟨i.val, by simp⟩
  let j' : Fin axes.toList.length := ⟨j.val, by simp⟩
  have hij' : axes.toList[i'.val] = axes.toList[j'.val] := by
    simpa [axesFunction, Vector.get, i', j'] using hij
  have heq : i' = j' := hList hij'
  apply Fin.ext
  simpa [i', j'] using congrArg Fin.val heq

/-- 入力軸が transpose 後のどの出力軸にあるかを計算する。 -/
private def inverseAxis
    {rank : Nat}
    (axes : Vector (Fin rank) rank)
    (hPermutation : IsAxisPermutation axes)
    (sourceAxis : Fin rank) :
    { outputAxis : Fin rank // axes.get outputAxis = sourceAxis } :=
  let axfunc := axesFunction axes
  let found := Fin.find (fun outputAxis => axfunc outputAxis = sourceAxis)
  have hSurjective :=
    (axesFunction_injective axes hPermutation).bijective_of_finite.2
  have hSome : found.isSome = true := by
    apply Option.isSome_iff_ne_none.mpr
    intro hNone
    obtain ⟨outputAxis, hAxis⟩ := hSurjective sourceAxis
    exact (Fin.find_eq_none_iff.mp hNone outputAxis) hAxis
  let outputAxis := found.get hSome
  have hFound :
      outputAxis ∈ Fin.find (fun outputAxis => axfunc outputAxis = sourceAxis) :=
    Option.get_mem hSome
  have hSpec : axfunc outputAxis = sourceAxis :=
    @Fin.find_spec rank (fun outputAxis => axfunc outputAxis = sourceAxis)
      (inferInstance) outputAxis hFound
  ⟨outputAxis, hSpec⟩

/--
指定された軸順列で Tensor を転置する。

`axes[i]` は、出力の第 `i` 軸に入力のどの軸を置くかを表す。
現在の Tensor は stride を持たないため、結果は row-major の新しい Array として
並べ直す。
-/
def transpose
    {α : Type}
    {rank : Nat}
    {shape : Vector Nat rank}
    (tensor : Tensor α shape)
    (axes : Vector (Fin rank) rank)
    (hPermutation : IsAxisPermutation axes) :
    Tensor α (transposeShape shape axes) :=
  let resultShape := transposeShape shape axes
  let inverse := inverseAxis axes hPermutation
  let data := Array.ofFn fun flatIndex : Fin (shapeSize resultShape) =>
    let resultIndex :=
      to_multi_index resultShape flatIndex flatIndex.isLt
    let sourceValues := Vector.ofFn fun sourceAxis =>
      resultIndex.values.get (inverse sourceAxis).val
    have sourceValid : index_in_bounds shape sourceValues := by
      intro sourceAxis
      have hResultValid :=
        resultIndex.isValid
          (inverse sourceAxis).val
      have hAxis := (inverse sourceAxis).property
      simpa [inverse, resultShape, transposeShape, sourceValues, hAxis] using hResultValid
    let sourceIndex : Index shape := {
      values := sourceValues
      isValid := sourceValid
    }
    tensor[sourceIndex]
  have hsize : data.size = shapeSize resultShape := by simp [data]
  { data := data, hsize := hsize }

private def manipulationExample : Tensor Nat #v[2, 3, 4] where
  data := Array.range 24
  hsize := by decide

private def reshapeExample : Tensor Nat #v[24, 1] :=
  reshape manipulationExample #v[24, 1] (by decide)

example : reshapeExample.data = Array.range 24 := by decide

private def transposeExample : Tensor Nat #v[3, 2, 4] :=
  transpose manipulationExample
    (#v[1, 0, 2] : Vector (Fin 3) 3) (by decide)

example :
    transposeExample.data =
      #[0, 1, 2, 3, 12, 13, 14, 15,
        4, 5, 6, 7, 16, 17, 18, 19,
        8, 9, 10, 11, 20, 21, 22, 23] := by
  decide

private def emptyTransposeExample : Tensor Nat #v[0, 2] where
  data := #[]
  hsize := by decide

example :
    (transpose emptyTransposeExample
      (#v[1, 0] : Vector (Fin 2) 2) (by decide)).data = #[] := by
  decide

end DL.Tensor
