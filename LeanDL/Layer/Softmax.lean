import LeanDL.Layer.Basic
import LeanDL.Tensor.Elementwise
import LeanDL.Tensor.Manipulation
import LeanDL.Tensor.Matmul
import LeanDL.Tensor.Reduction

namespace DL.Softmax

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

private def vectorAsColumn
    {α : Type} {rows : Nat}
    (tensor : Tensor α #v[rows]) : Tensor α #v[rows, 1] :=
  Tensor.reshape tensor _ (by simp [shapeSize])

private def batchedVectorAsRowMatrix
    {α : Type} {batchSize features : Nat}
    (tensor : BatchedTensor α #v[features] batchSize) :
    Tensor α (#v[batchSize] ++ #v[1, features]) :=
  Tensor.reshape tensor _ (by
    unfold shapeSize
    rw [Vector.foldl_append, Vector.foldl_append]
    simp)

private def batchedVectorAsColumnMatrix
    {α : Type} {batchSize features : Nat}
    (tensor : BatchedTensor α #v[features] batchSize) :
    Tensor α (#v[batchSize] ++ #v[features, 1]) :=
  Tensor.reshape tensor _ (by
    unfold shapeSize
    rw [Vector.foldl_append, Vector.foldl_append]
    simp)

private def batchedScalarMatrixAsColumn
    {α : Type} {batchSize : Nat}
    (tensor : Tensor α (#v[batchSize] ++ #v[1, 1])) :
    Tensor α #v[batchSize, 1] :=
  Tensor.reshape tensor _ (by
    unfold shapeSize
    rw [Vector.foldl_append]
    simp)

private def zipWithMatrixColumn
    {α β γ : Type} {rows cols : Nat}
    (matrix : Tensor α #v[rows, cols])
    (column : Tensor β #v[rows, 1])
    (f : α → β → γ) : Tensor γ #v[rows, cols] :=
  Tensor.zipWith matrix column f (Tensor.broadcast_matrix_column rows cols)

private def negativeInfinity : Float :=
  -1 / 0

private def maximum (left right : Float) : Float :=
  if left < right then right else left

/-- Softmaxのbackwardに必要な、batch size付き出力cache。 -/
private structure Cache (features : Nat) where
  batchSize : Nat
  output : BatchedTensor Float #v[features] batchSize

/-- parameterを持たず、直近のSoftmax出力だけを保持するstate。 -/
private structure State (features : Nat) where
  cache : Option (Cache features)

private def cacheBatchSize
    {features : Nat} (state : State features) : Option Nat :=
  state.cache.map Cache.batchSize

private def forwardState
    {features batchSize : Nat}
    (input : BatchedTensor Float #v[features] batchSize) :
    StateM (State features) (BatchedTensor Float #v[features] batchSize) :=
  fun state =>
    let inputMatrix := batchedVectorAsMatrix input
    let rowMaximums := Tensor.foldAxis
      inputMatrix (1 : Fin 2) negativeInfinity maximum
    let centered := zipWithMatrixColumn inputMatrix (vectorAsColumn rowMaximums)
      (· - ·)
    let exponentials := Tensor.map centered Float.exp
    let rowSums := Tensor.foldAxis exponentials (1 : Fin 2) 0 (· + ·)
    let outputMatrix := zipWithMatrixColumn exponentials (vectorAsColumn rowSums)
      (· / ·)
    let output : BatchedTensor Float #v[features] batchSize :=
      matrixAsBatchedVector outputMatrix
    let cache : Cache features := { batchSize, output }
    (output, { state with cache := some cache })

private def backwardState
    {features batchSize : Nat}
    (outputGradient : BatchedTensor Float #v[features] batchSize)
    (state : State features)
    (hBatch : cacheBatchSize state = some batchSize) :
    BatchedTensor Float #v[features] batchSize × State features :=
  match hCache : state.cache with
  | none => by
      simp [cacheBatchSize, hCache] at hBatch
  | some cache =>
      have hCacheBatch : cache.batchSize = batchSize := by
        simpa [cacheBatchSize, hCache] using hBatch
      let output : BatchedTensor Float #v[features] batchSize :=
        hCacheBatch ▸ cache.output
      let outputMatrix := batchedVectorAsMatrix output
      let outputGradientMatrix := batchedVectorAsMatrix outputGradient
      have hBatchBroadcast :
          Tensor.broadcast #v[batchSize] #v[batchSize] =
            some #v[batchSize] := by
        simpa [Tensor.broadcastSelfShape, Vector.cast] using
          (Tensor.broadcast_self #v[batchSize])
      let rowDotProductsWithMatrixDimensions := Tensor.matmul
        (resultBatch := #v[batchSize]) false false
        (batchedVectorAsRowMatrix outputGradient)
        (batchedVectorAsColumnMatrix output)
        (· * ·) (· + ·) 0 (by rfl) hBatchBroadcast
      let rowDotProducts :=
        batchedScalarMatrixAsColumn rowDotProductsWithMatrixDimensions
      let centeredGradient := zipWithMatrixColumn
        outputGradientMatrix rowDotProducts (· - ·)
      let inputGradientMatrix := Tensor.zipWithSame
        outputMatrix centeredGradient (· * ·)
      let inputGradient : BatchedTensor Float #v[features] batchSize :=
        matrixAsBatchedVector inputGradientMatrix
      (inputGradient, state)

/--
一サンプルの `#[features]` に対し、feature軸へSoftmaxを適用するFloat専用Layerを作る。

forwardでは各batch行の最大値を引いてから指数関数を適用する。
-/
def new {features : Nat} : Layer Float #v[features] #v[features] where
  State := State features
  state := { cache := none }
  cachedBatchSizeState := cacheBatchSize
  forwardState := forwardState
  forwardCachesBatch := by
    intro batchSize input state
    simp [forwardState, cacheBatchSize]
  backwardState := backwardState

-- ここから先は検証用の example と、それに付随する private 定義。

private def testInput : BatchedTensor Float #v[2] 2 where
  data := #[1000, 1000, 0, 0]
  hsize := by decide

private def testOutputGradient : BatchedTensor Float #v[2] 2 where
  data := #[1, 0, 0, 1]
  hsize := by decide

private theorem forwardAndBackwardValues :
    let layer : Layer Float #v[2] #v[2] := new
    let forwardResult := layer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward layer testInput)
    forwardResult.1.data.map Float.toBits =
        #[0.5, 0.5, 0.5, 0.5].map Float.toBits ∧
      backwardResult.1.data.map Float.toBits =
        #[0.25, -0.25, -0.25, 0.25].map Float.toBits := by
  native_decide

end DL.Softmax
