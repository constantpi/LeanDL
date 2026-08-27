import LeanDL.Layer.Basic

namespace DL.Model

/--
Layerを直列に接続したモデル。

隣接するLayerのoutput shapeとinput shapeは、共通の `hiddenShape` indexによって
型レベルで一致する。`nil` はshapeを変えない恒等モデルを表す。
-/
inductive Sequential (α : Type) :
    {inputRank outputRank : Nat} →
    Vector Nat inputRank → Vector Nat outputRank → Type 1
  | nil {rank : Nat} {shape : Vector Nat rank} :
      Sequential α shape shape
  | cons
      {inputRank hiddenRank outputRank : Nat}
      {inputShape : Vector Nat inputRank}
      {hiddenShape : Vector Nat hiddenRank}
      {outputShape : Vector Nat outputRank}
      (head : Layer α inputShape hiddenShape)
      (tail : Sequential α hiddenShape outputShape) :
      Sequential α inputShape outputShape

/-- 1次元入力から1次元出力への直列モデル。 -/
abbrev VectorModel (α : Type) (inputSize outputSize : Nat) :=
  Sequential α #v[inputSize] #v[outputSize]

namespace Sequential

/-- model内のすべてのLayerが、指定されたbatch sizeのcacheを持つこと。 -/
def CachesBatch
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank} :
    Sequential α inputShape outputShape → Nat → Prop
  | .nil, _ => True
  | .cons head tail, batchSize =>
      head.cachedBatchSize = some batchSize ∧ tail.CachesBatch batchSize

/--
modelのforwardを先頭のLayerから順番に実行する。

結果とともに、各Layerが更新済みcacheを持つ新しいmodelを返す。
-/
def forward
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank}
    (model : Sequential α inputShape outputShape)
    {batchSize : Nat}
    (input : BatchedTensor α inputShape batchSize) :
    BatchedTensor α outputShape batchSize ×
      Sequential α inputShape outputShape :=
  match model with
  | .nil => (input, .nil)
  | .cons head tail =>
      let headResult := head.forward input
      let tailResult := tail.forward headResult.1
      (tailResult.1, .cons headResult.2 tailResult.2)

/-- forward後のmodelでは、すべてのLayerが入力と同じbatch sizeをcacheする。 -/
theorem cachesBatch_forward
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank}
    (model : Sequential α inputShape outputShape)
    {batchSize : Nat}
    (input : BatchedTensor α inputShape batchSize) :
    (model.forward input).2.CachesBatch batchSize := by
  induction model with
  | nil =>
      simp [forward, CachesBatch]
  | cons head tail ih =>
      simp only [forward, CachesBatch]
      exact ⟨Layer.cachedBatchSize_forward head input, ih (head.forward input).1⟩

/--
modelのbackwardを末尾のLayerから逆順に実行する。

`hBatch` により、すべてのLayerのcacheとoutput gradientのbatch sizeが一致する。
-/
def backward
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank}
    (model : Sequential α inputShape outputShape)
    {batchSize : Nat}
    (outputGradient : BatchedTensor α outputShape batchSize)
    (hBatch : model.CachesBatch batchSize) :
    BatchedTensor α inputShape batchSize ×
      Sequential α inputShape outputShape :=
  match model with
  | .nil => (outputGradient, .nil)
  | .cons head tail =>
      let tailResult := tail.backward outputGradient hBatch.2
      let headResult := head.backward tailResult.1 hBatch.1
      (headResult.1, .cons headResult.2 tailResult.2)

/-- shapeが接続可能な2つの直列モデルを連結する。 -/
def append
    {α : Type}
    {inputRank middleRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {middleShape : Vector Nat middleRank}
    {outputShape : Vector Nat outputRank}
    (first : Sequential α inputShape middleShape)
    (second : Sequential α middleShape outputShape) :
    Sequential α inputShape outputShape :=
  match first with
  | .nil => second
  | .cons head tail => .cons head (tail.append second)

end Sequential

-- ここから先は検証用の example と、それに付随する private 定義。

private structure TestState where
  cachedBatchSize : Option Nat := none

private def testLayer : Layer Int #v[3] #v[3] where
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

private def testModel : VectorModel Int 3 3 :=
  .cons testLayer (.cons testLayer .nil)

private def testInput : BatchedTensor Int #v[3] 2 where
  data := #[-1, 2, 3, 4, -5, 6]
  hsize := by decide

private def testOutputGradient : BatchedTensor Int #v[3] 2 :=
  Tensor.fill (#v[2] ++ #v[3]) 1

private theorem forwardAndBackwardValues :
    let forwardResult := testModel.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Sequential.cachesBatch_forward testModel testInput)
    forwardResult.1.data = #[-1, 2, 3, 4, -5, 6] ∧
      backwardResult.1.data = #[1, 1, 1, 1, 1, 1] := by
  native_decide

private theorem forwardCachesBatchInEveryLayer :
    (testModel.forward testInput).2.CachesBatch 2 := by
  exact Sequential.cachesBatch_forward testModel testInput

end DL.Model
