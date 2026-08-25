import LeanDL.Layer.Basic
import LeanDL.Tensor.Elementwise

namespace DL.Activation

/-- elementwise activationのbackwardに必要な、batch size付き出力cache。 -/
private structure Cache
    (α : Type) {rank : Nat} (sampleShape : Vector Nat rank) where
  batchSize : Nat
  output : BatchedTensor α sampleShape batchSize

/-- parameterを持たず、直近のforward出力だけを保持するactivation state。 -/
private structure State
    (α : Type) {rank : Nat} (sampleShape : Vector Nat rank) where
  cache : Option (Cache α sampleShape)

private def cacheBatchSize
    {α : Type} {rank : Nat} {sampleShape : Vector Nat rank}
    (state : State α sampleShape) : Option Nat :=
  state.cache.map Cache.batchSize

private def forwardState
    {α : Type} {rank : Nat} {sampleShape : Vector Nat rank}
    (forwardElement : α → α)
    {batchSize : Nat}
    (input : BatchedTensor α sampleShape batchSize) :
    StateM (State α sampleShape) (BatchedTensor α sampleShape batchSize) :=
  fun state =>
    let output := input.map forwardElement
    let cache : Cache α sampleShape := { batchSize, output }
    (output, { state with cache := some cache })

private def backwardState
    {α : Type} {rank : Nat} {sampleShape : Vector Nat rank}
    (backwardElement : α → α → α)
    {batchSize : Nat}
    (outputGradient : BatchedTensor α sampleShape batchSize)
    (state : State α sampleShape)
    (hBatch : cacheBatchSize state = some batchSize) :
    BatchedTensor α sampleShape batchSize × State α sampleShape :=
  match hCache : state.cache with
  | none => by
      simp [cacheBatchSize, hCache] at hBatch
  | some cache =>
      have hCacheBatch : cache.batchSize = batchSize := by
        simpa [cacheBatchSize, hCache] using hBatch
      let output : BatchedTensor α sampleShape batchSize :=
        hCacheBatch ▸ cache.output
      let inputGradient :=
        Tensor.zipWithSame output outputGradient backwardElement
      (inputGradient, state)

/--
要素ごとのforward関数と、forward出力から入力勾配を計算する関数でactivation Layerを作る。

`backwardElement output outputGradient` は、その要素に対応するinput gradientを返す。
Layerはparameterを持たず、forwardのたびに直前の出力cacheを上書きする。
-/
def newFromOutput
    {α : Type} {rank : Nat} {sampleShape : Vector Nat rank}
    (forwardElement : α → α)
    (backwardElement : α → α → α) -- output → outputGradient → inputGradient
    : Layer α sampleShape sampleShape where
  State := State α sampleShape
  state := { cache := none }
  cachedBatchSizeState := cacheBatchSize
  forwardState := forwardState forwardElement
  forwardCachesBatch := by
    intro batchSize input state
    simp [forwardState, cacheBatchSize]
  backwardState := backwardState backwardElement

-- ここから先は検証用の example と、それに付随する private 定義。

private def testLayer : Layer Nat #v[2] #v[2] :=
  newFromOutput (· + 1) fun output outputGradient => output * outputGradient

private def testInput : BatchedTensor Nat #v[2] 2 where
  data := #[1, 2, 3, 4]
  hsize := by decide

private def testOutputGradient : BatchedTensor Nat #v[2] 2 :=
  Tensor.fill (#v[2] ++ #v[2]) 2

private theorem forwardAndBackwardValues :
    let forwardResult := testLayer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward testLayer testInput)
    forwardResult.1.data = #[2, 3, 4, 5] ∧
      backwardResult.1.data = #[4, 6, 8, 10] := by
  native_decide

end DL.Activation
