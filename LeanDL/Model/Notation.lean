import LeanDL.Model.Sequential

namespace DL.Model

/--
Layerの列から直列モデルを構築する。

末尾の `Sequential.nil` はマクロによって補われる。
-/
macro "model![" layers:term,* "]" : term => do
  let mut model ← `(Sequential.nil)
  for layer in layers.getElems.reverse do
    model ← `(Sequential.cons $layer $model)
  return model

/-- 接続可能な2つの直列モデルを連結する `>>>` 記法。 -/
instance
    {α : Type}
    {inputRank middleRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {middleShape : Vector Nat middleRank}
    {outputShape : Vector Nat outputRank} :
    HShiftRight
      (Sequential α inputShape middleShape)
      (Sequential α middleShape outputShape)
      (Sequential α inputShape outputShape) where
  hShiftRight := Sequential.append

/-- 接続可能な2つのLayerから直列モデルを構築する `>>>` 記法。 -/
instance
    {α : Type}
    {inputRank middleRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {middleShape : Vector Nat middleRank}
    {outputShape : Vector Nat outputRank} :
    HShiftRight
      (Layer α inputShape middleShape)
      (Layer α middleShape outputShape)
      (Sequential α inputShape outputShape) where
  hShiftRight first second := .cons first (.cons second .nil)

/-- 直列モデルの末尾に、接続可能なLayerを追加する `>>>` 記法。 -/
instance
    {α : Type}
    {inputRank middleRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {middleShape : Vector Nat middleRank}
    {outputShape : Vector Nat outputRank} :
    HShiftRight
      (Sequential α inputShape middleShape)
      (Layer α middleShape outputShape)
      (Sequential α inputShape outputShape) where
  hShiftRight model layer := model.append (.cons layer .nil)

/-- Layerの後ろに、接続可能な直列モデルを追加する `>>>` 記法。 -/
instance
    {α : Type}
    {inputRank middleRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {middleShape : Vector Nat middleRank}
    {outputShape : Vector Nat outputRank} :
    HShiftRight
      (Layer α inputShape middleShape)
      (Sequential α middleShape outputShape)
      (Sequential α inputShape outputShape) where
  hShiftRight layer model := .cons layer model

-- ここから先は検証用の example と、それに付随する private 定義。

private structure TestState where
  cachedBatchSize : Option Nat := none

private def testLayer : Layer Int #v[2] #v[2] where
  State := TestState
  state := {}
  cachedBatchSizeState := TestState.cachedBatchSize
  forwardState := fun {batchSize} input state =>
    (input, { state with cachedBatchSize := some batchSize })
  forwardCachesBatch := by
    intro batchSize input state
    rfl
  backwardState := fun outputGradient state _hBatch =>
    (outputGradient, state)

private def testInput : BatchedTensor Int #v[2] 2 where
  data := #[1, 2, 3, 4]
  hsize := by decide

private def firstModel : VectorModel Int 2 2 :=
  model![testLayer, testLayer]

private def secondModel : VectorModel Int 2 2 :=
  model![testLayer]

private def operatorModel : VectorModel Int 2 2 :=
  testLayer >>> testLayer >>> testLayer

example : VectorModel Int 2 2 :=
  model![]

example :
    (firstModel.forward testInput).1.data = #[1, 2, 3, 4] := by
  native_decide

example :
    let model := operatorModel >>> secondModel
    (model.forward testInput).2.CachesBatch 2 := by
  exact Sequential.cachesBatch_forward (operatorModel >>> secondModel) testInput

end DL.Model
