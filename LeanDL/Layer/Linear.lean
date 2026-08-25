import LeanDL.Layer.Basic
import LeanDL.Tensor.Matrix
import LeanDL.Tensor.Reduction

namespace DL
namespace Linear

private def batchedVectorAsMatrix
    {α : Type} {batchSize features : Nat}
    (tensor : BatchedTensor α #v[features] batchSize) :
    Tensor α #v[batchSize, features] :=
  Tensor.reshape tensor _ (by
    unfold shapeSize
    rw [Vector.foldl_append]
    simp)

private def matrixAsBatchedVector
    {α : Type} {batchSize features : Nat}
    (tensor : Tensor α #v[batchSize, features]) :
    BatchedTensor α #v[features] batchSize :=
  Tensor.reshape tensor _ (by
    unfold shapeSize
    rw [Vector.foldl_append]
    simp)

/-- Linear layer の backward に必要な、batch size 付き input cache。 -/
private structure Cache (α : Type) (inFeatures : Nat) where
  batchSize : Nat
  input : BatchedTensor α #v[inFeatures] batchSize

/-- parameter、単一の forward cache、蓄積 parameter gradient。 -/
private structure State
    (α : Type) (inFeatures outFeatures : Nat) where
  weight : Tensor α #v[inFeatures, outFeatures]
  bias : Tensor α #v[outFeatures]
  cache : Option (Cache α inFeatures)
  weightGradient : Tensor α #v[inFeatures, outFeatures]
  biasGradient : Tensor α #v[outFeatures]

private def cacheBatchSize
    {α : Type} {inFeatures outFeatures : Nat}
    (state : State α inFeatures outFeatures) : Option Nat :=
  state.cache.map Cache.batchSize

private def forwardState
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures batchSize : Nat}
    (input : BatchedTensor α #v[inFeatures] batchSize) :
    StateM (State α inFeatures outFeatures)
      (BatchedTensor α #v[outFeatures] batchSize) :=
  fun state =>
    let product := Tensor.matmul2D false false
      (batchedVectorAsMatrix input) state.weight
      (· * ·) (· + ·) 0 (by rfl)
    let outputMatrix := Tensor.zipWithMatrixVector product state.bias (· + ·)
    let output : BatchedTensor α #v[outFeatures] batchSize :=
      matrixAsBatchedVector outputMatrix
    let cache : Cache α inFeatures := { batchSize, input }
    (output, { state with cache := some cache })

private def backwardState
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures batchSize : Nat}
    (outputGradient : BatchedTensor α #v[outFeatures] batchSize)
    (state : State α inFeatures outFeatures)
    (hBatch : cacheBatchSize state = some batchSize) :
    BatchedTensor α #v[inFeatures] batchSize ×
      State α inFeatures outFeatures :=
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
        outputGradientMatrix state.weight
        (· * ·) (· + ·) 0 (by rfl)
      let inputGradient : BatchedTensor α #v[inFeatures] batchSize :=
        matrixAsBatchedVector inputGradientMatrix
      let currentWeightGradient := Tensor.matmul2D true false
        (batchedVectorAsMatrix input) outputGradientMatrix
        (· * ·) (· + ·) 0 (by rfl)
      let accumulatedWeightGradient := Tensor.zipWithSame
        state.weightGradient currentWeightGradient (· + ·)
      let currentBiasGradient := Tensor.foldAxis
        outputGradientMatrix (0 : Fin 2) 0 (· + ·)
      let accumulatedBiasGradient := Tensor.zipWithSame
        state.biasGradient currentBiasGradient (· + ·)
      (inputGradient, { state with
        weightGradient := accumulatedWeightGradient
        biasGradient := accumulatedBiasGradient
      })

/--
初期 parameter を受け取り、#[inFeatures] → #[outFeatures] の Linear layer を作る。

weight の shape は #[inFeatures, outFeatures]。parameter gradient は zero で
初期化される。forward は以前の cache を上書きし、backward は weight/bias を
変更せず gradient のみを蓄積する。
-/
def new
    {α : Type} [Add α] [Mul α] [OfNat α 0]
    {inFeatures outFeatures : Nat}
    (weight : Tensor α #v[inFeatures, outFeatures])
    (bias : Tensor α #v[outFeatures]) :
    Layer α #v[inFeatures] #v[outFeatures] where
  State := State α inFeatures outFeatures
  state := {
    weight
    bias
    cache := none
    weightGradient := Tensor.fill #v[inFeatures, outFeatures] 0
    biasGradient := Tensor.fill #v[outFeatures] 0
  }
  cachedBatchSizeState := cacheBatchSize
  forwardState := forwardState
  forwardCachesBatch := by -- forwardState が返す batch size は入力と一致する
    intro batchSize input state
    simp [forwardState, cacheBatchSize]
  backwardState := backwardState

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

private def testState : State Nat 2 2 where
  weight := testWeight
  bias := testBias
  cache := none
  weightGradient := Tensor.fill #v[2, 2] 10
  biasGradient := Tensor.fill #v[2] 10

private theorem forwardAndBackwardValues :
    let forwardResult := forwardState testInput testState
    let backwardResult :=
      backwardState testOutputGradient forwardResult.2 (by rfl)
    forwardResult.1.data = #[17, 30, 25, 42] ∧
      backwardResult.1.data = #[3, 7, 3, 7] ∧
      backwardResult.2.weightGradient.data = #[14, 14, 16, 16] ∧
      backwardResult.2.biasGradient.data = #[12, 12] := by
  native_decide

private theorem publicBackwardAcceptsForwardBatchProof :
    let layer := new testWeight testBias
    let forwardResult := layer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward layer testInput)
    backwardResult.1.data = #[3, 7, 3, 7] := by
  native_decide

private theorem forwardOverwritesCachedBatch :
    let layer := new testWeight testBias
    let firstResult := layer.forward (Tensor.fill (#v[1] ++ #v[2]) 0)
    let secondResult := firstResult.2.forward testInput
    secondResult.2.cachedBatchSize = some 2 := by
  exact Layer.cachedBatchSize_forward _ _

end Linear
end DL
