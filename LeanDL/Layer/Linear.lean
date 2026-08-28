import LeanDL.Layer.Basic
import LeanDL.Tensor.Matrix
import LeanDL.Tensor.Reduction
import LeanDL.Tensor.Tactics

namespace DL
namespace Linear

private def batchedVectorAsMatrix
    {α : Type} {batchSize features : Nat}
    (tensor : BatchedTensor α #v[features] batchSize) :
    Tensor α #v[batchSize, features] :=
  Tensor.reshape tensor _ (by shape_simp)

private def matrixAsBatchedVector
    {α : Type} {batchSize features : Nat}
    (tensor : Tensor α #v[batchSize, features]) :
    BatchedTensor α #v[features] batchSize :=
  Tensor.reshape tensor _ (by shape_simp)

/-- Linear layer の backward に必要な、batch size 付き input cache。 -/
private structure Cache (α : Type) (inFeatures : Nat) where
  batchSize : Nat
  input : BatchedTensor α #v[inFeatures] batchSize

/-- parameter、単一の forward cache、蓄積 parameter gradient。 -/
private structure State
    (α : Type) (inFeatures outFeatures : Nat)
    (WeightOptimizerState BiasOptimizerState : Type) where
  weight : Parameter α #v[inFeatures, outFeatures] WeightOptimizerState
  bias : Parameter α #v[outFeatures] BiasOptimizerState
  cache : Option (Cache α inFeatures)

private def cacheBatchSize
    {α : Type} {inFeatures outFeatures : Nat}
    {WeightOptimizerState BiasOptimizerState : Type}
    (state : State α inFeatures outFeatures
      WeightOptimizerState BiasOptimizerState) : Option Nat :=
  state.cache.map Cache.batchSize

private def forwardState
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures batchSize : Nat}
    {WeightOptimizerState BiasOptimizerState : Type}
    (input : BatchedTensor α #v[inFeatures] batchSize) :
    StateM (State α inFeatures outFeatures
      WeightOptimizerState BiasOptimizerState)
      (BatchedTensor α #v[outFeatures] batchSize) :=
  fun state =>
    let product := Tensor.matmul2D false false
      (batchedVectorAsMatrix input) state.weight.value
      (· * ·) (· + ·) 0 (by rfl)
    let outputMatrix := Tensor.zipWithMatrixVector product state.bias.value (· + ·)
    let output : BatchedTensor α #v[outFeatures] batchSize :=
      matrixAsBatchedVector outputMatrix
    let cache : Cache α inFeatures := { batchSize, input }
    (output, { state with cache := some cache })

private def backwardState
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures batchSize : Nat}
    {WeightOptimizerState BiasOptimizerState : Type}
    (outputGradient : BatchedTensor α #v[outFeatures] batchSize)
    (state : State α inFeatures outFeatures
      WeightOptimizerState BiasOptimizerState)
    (hBatch : cacheBatchSize state = some batchSize) :
    BatchedTensor α #v[inFeatures] batchSize ×
      State α inFeatures outFeatures
        WeightOptimizerState BiasOptimizerState :=
  match hCache : state.cache with
  | none => by
      -- 矛盾しているので、ここに到達することはない。
      simp [cacheBatchSize, hCache] at hBatch
  | some cache =>
      have hCacheBatch : cache.batchSize = batchSize := by
        simpa [cacheBatchSize, hCache] using hBatch
      let input : BatchedTensor α #v[inFeatures] batchSize :=
        hCacheBatch ▸ cache.input
      let outputGradientMatrix := batchedVectorAsMatrix outputGradient
      let inputGradientMatrix := Tensor.matmul2D false true
        outputGradientMatrix state.weight.value
        (· * ·) (· + ·) 0 (by rfl)
      let inputGradient : BatchedTensor α #v[inFeatures] batchSize :=
        matrixAsBatchedVector inputGradientMatrix
      let currentWeightGradient := Tensor.matmul2D true false
        (batchedVectorAsMatrix input) outputGradientMatrix
        (· * ·) (· + ·) 0 (by rfl)
      let weight := state.weight.accumulate currentWeightGradient
      let currentBiasGradient := Tensor.foldAxis
        outputGradientMatrix (0 : Fin 2) 0 (· + ·)
      let bias := state.bias.accumulate currentBiasGradient
      (inputGradient, { state with
        weight
        bias
      })

private def stepState
    {α : Type} [OfNat α 0]
    {inFeatures outFeatures : Nat}
    {WeightOptimizerState BiasOptimizerState : Type} :
    StateM (State α inFeatures outFeatures
      WeightOptimizerState BiasOptimizerState) Unit :=
  fun state =>
    ((), {
      weight := state.weight.step
      bias := state.bias.step
      -- parameter更新前のforward cacheを使ったbackwardを防ぐ。
      cache := none
    })

private def zeroGradState
    {α : Type} [OfNat α 0]
    {inFeatures outFeatures : Nat}
    {WeightOptimizerState BiasOptimizerState : Type} :
    StateM (State α inFeatures outFeatures
      WeightOptimizerState BiasOptimizerState) Unit :=
  fun state =>
    ((), { state with
      weight := state.weight.zeroGrad
      bias := state.bias.zeroGrad
    })

/--
初期 parameter を受け取り、#[inFeatures] → #[outFeatures] の Linear layer を作る。

weight の shape は #[inFeatures, outFeatures]。weightとbiasにはそれぞれshapeが
一致するOptimizerを渡す。parameter gradientはzeroで初期化され、forwardは以前の
cacheを上書きし、backwardはweight/biasを変更せずgradientのみを蓄積する。
`Layer.step` は両parameterを更新してgradientをzeroに戻し、古いcacheを破棄する。
-/
def new
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures : Nat}
    {WeightOptimizerState BiasOptimizerState : Type}
    (weight : Tensor α #v[inFeatures, outFeatures])
    (bias : Tensor α #v[outFeatures])
    (weightOptimizer : Optimizer α #v[inFeatures, outFeatures]
      WeightOptimizerState)
    (biasOptimizer : Optimizer α #v[outFeatures] BiasOptimizerState) :
    Layer α #v[inFeatures] #v[outFeatures] where
  State := State α inFeatures outFeatures
    WeightOptimizerState BiasOptimizerState
  state := {
    weight := Parameter.new weight weightOptimizer
    bias := Parameter.new bias biasOptimizer
    cache := none
  }
  cachedBatchSizeState := cacheBatchSize
  forwardState := forwardState
  forwardCachesBatch := by -- forwardState が返す batch size は入力と一致する
    intro batchSize input state
    simp [forwardState, cacheBatchSize]
  backwardState := backwardState
  stepState := stepState
  zeroGradState := zeroGradState

-- ここから先は検証用の example と、それに付随する private 定義。

private def testWeight : Tensor Nat #v[2, 2] where
  data := #[1, 2, 3, 4]
  hsize := by decide

private def testBias : Tensor Nat #v[2] where
  data := #[10, 20]
  hsize := by decide

private def testInput : BatchedTensor Nat #v[2] 2 where
  data := #[1, 2, 3, 4]
  hsize := by decide

private def testOutputGradient : BatchedTensor Nat #v[2] 2 :=
  Tensor.fill (#v[2] ++ #v[2]) 1

private def testWeightOptimizer : Optimizer Nat #v[2, 2] Unit where
  state := ()
  stepState := fun value _gradient state => (value, state)

private def testBiasOptimizer : Optimizer Nat #v[2] Unit where
  state := ()
  stepState := fun value _gradient state => (value, state)

private def testState : State Nat 2 2 Unit Unit where
  weight := {
    value := testWeight
    accumulatedGradient := Tensor.fill #v[2, 2] 10
    optimizer := testWeightOptimizer
  }
  bias := {
    value := testBias
    accumulatedGradient := Tensor.fill #v[2] 10
    optimizer := testBiasOptimizer
  }
  cache := none

private theorem forwardAndBackwardValues :
    let forwardResult := forwardState testInput testState
    let backwardResult :=
      backwardState testOutputGradient forwardResult.2 (by rfl)
    forwardResult.1.data = #[17, 30, 25, 42] ∧
      backwardResult.1.data = #[3, 7, 3, 7] ∧
      backwardResult.2.weight.accumulatedGradient.data = #[14, 14, 16, 16] ∧
      backwardResult.2.bias.accumulatedGradient.data = #[12, 12] := by
  native_decide

private theorem publicBackwardAcceptsForwardBatchProof :
    let layer := new testWeight testBias testWeightOptimizer testBiasOptimizer
    let forwardResult := layer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward layer testInput)
    backwardResult.1.data = #[3, 7, 3, 7] := by
  native_decide

private theorem forwardOverwritesCachedBatch :
    let layer := new testWeight testBias testWeightOptimizer testBiasOptimizer
    let firstResult := layer.forward (Tensor.fill (#v[1] ++ #v[2]) 0)
    let secondResult := firstResult.2.forward testInput
    secondResult.2.cachedBatchSize = some 2 := by
  exact Layer.cachedBatchSize_forward _ _

private theorem stepInvalidatesForwardCache :
    let layer := new testWeight testBias testWeightOptimizer testBiasOptimizer
    let forwardResult := layer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward layer testInput)
    backwardResult.2.step.cachedBatchSize = none := by
  native_decide

end Linear
end DL
