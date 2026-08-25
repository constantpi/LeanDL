import LeanDL.Tensor.Basic

namespace DL

/--
一サンプル分の `sampleShape` の前に、実行時に選ぶ batch 次元を付けた Tensor。

`batchSize` は `Layer` 自体の型には含まれず、`forward` / `backward` を呼ぶたびに
選ばれる。
-/
abbrev BatchedTensor
    (α : Type) {rank : Nat} (sampleShape : Vector Nat rank)
    (batchSize : Nat) : Type :=
  Tensor α (#v[batchSize] ++ sampleShape)

/--
shape 付き Tensor を変換する、状態を持つ Layer の共通インターフェース。

`inputShape` と `outputShape` は一サンプル分の shape である。そのため、Layer の型は
batch size に依存せず、input/output の rank が異なっていてもよい。

Layer 固有の parameter、forward cache、蓄積済み parameter gradient は、実装側が
`State` に格納する。`forwardState` は backward に必要な値を state に保存し、
`backwardState` は cache を利用して parameter gradient を既存値へ加算することを
想定している。parameter の更新はこのインターフェースには含めず、backward から
分離する。

`StateM` を使うことで状態の受け渡しを明示しつつ、利用側は `Layer.forward` と
`Layer.backward` が返す更新済み Layer を次の呼び出しへ渡すだけでよい。
-/
structure Layer
    (α : Type)
    {inputRank outputRank : Nat}
    (inputShape : Vector Nat inputRank)
    (outputShape : Vector Nat outputRank) where
  /-- Layer ごとに異なる、外部から抽象化された内部状態。 -/
  State : Type
  /-- parameter、cache、蓄積 gradient を含む現在の状態。 -/
  state : State
  /-- 出力を計算し、backward に必要な cache を state に保存する。 -/
  forwardState : {batchSize : Nat} →
    BatchedTensor α inputShape batchSize →
    StateM State (BatchedTensor α outputShape batchSize)
  /-- 入力 gradient を計算し、parameter gradient を state に蓄積する。 -/
  backwardState : {batchSize : Nat} →
    BatchedTensor α outputShape batchSize →
    StateM State (BatchedTensor α inputShape batchSize)

namespace Layer

/--
Layer の forward を実行する。

結果の第1要素が出力、第2要素が cache を保持した更新済み Layer である。
-/
def forward
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank}
    (layer : Layer α inputShape outputShape)
    {batchSize : Nat}
    (input : BatchedTensor α inputShape batchSize) :
    BatchedTensor α outputShape batchSize × Layer α inputShape outputShape :=
  let (output, nextState) := layer.forwardState input layer.state
  (output, { layer with state := nextState })

/--
Layer の backward を実行する。

結果の第1要素が入力 gradient、第2要素が parameter gradient を蓄積した更新済み
Layer である。parameter 自体の更新は行わない。
-/
def backward
    {α : Type}
    {inputRank outputRank : Nat}
    {inputShape : Vector Nat inputRank}
    {outputShape : Vector Nat outputRank}
    (layer : Layer α inputShape outputShape)
    {batchSize : Nat}
    (outputGradient : BatchedTensor α outputShape batchSize) :
    BatchedTensor α inputShape batchSize × Layer α inputShape outputShape :=
  let (inputGradient, nextState) :=
    layer.backwardState outputGradient layer.state
  (inputGradient, { layer with state := nextState })

end Layer

/-!
以下は abstraction の性質をコンパイル時に確認するための小さな例である。
実際のニューラルネットワーク Layer の実装ではない。
-/

private structure DummyState where
  cachedBatchSizes : List Nat := []
  accumulatedGradient : Nat := 0

/-- dummy Layer の rank の異なる二つの sample shape は要素数が等しい。 -/
private theorem dummyShapeSize (batchSize : Nat) :
    shapeSize (#v[batchSize] ++ #v[2]) =
      shapeSize (#v[batchSize] ++ #v[1, 2]) := by
  unfold shapeSize
  rw [Vector.foldl_append, Vector.foldl_append]
  simp

private def dummyForwardState
    {batchSize : Nat}
    (input : BatchedTensor Nat #v[2] batchSize) :
    StateM DummyState (BatchedTensor Nat #v[1, 2] batchSize) :=
  fun state =>
    let output : BatchedTensor Nat #v[1, 2] batchSize := {
      data := input.data
      hsize := input.hsize.trans (dummyShapeSize batchSize)
    }
    (output, { state with
      cachedBatchSizes := batchSize :: state.cachedBatchSizes })

private def dummyBackwardState
    {batchSize : Nat}
    (outputGradient : BatchedTensor Nat #v[1, 2] batchSize) :
    StateM DummyState (BatchedTensor Nat #v[2] batchSize) :=
  fun state =>
    let inputGradient : BatchedTensor Nat #v[2] batchSize := {
      data := outputGradient.data
      hsize := outputGradient.hsize.trans (dummyShapeSize batchSize).symm
    }
    -- cache と output gradient の両方を使って Dummy parameter gradient を作る。
    let cachedBatchSize := state.cachedBatchSizes.head?.getD 0
    let parameterGradient := outputGradient.data.size + cachedBatchSize
    let remainingCache := state.cachedBatchSizes.drop 1
    (inputGradient, {
      cachedBatchSizes := remainingCache
      accumulatedGradient := state.accumulatedGradient + parameterGradient
    })

/-- rank 1 の sample を、要素数を変えず rank 2 にするテスト用 Layer。 -/
private def dummyLayer : Layer Nat #v[2] #v[1, 2] where
  State := DummyState
  state := {}
  forwardState := dummyForwardState
  backwardState := dummyBackwardState

private def dummyBatchOne : BatchedTensor Nat #v[2] 1 :=
  Tensor.fill (#v[1] ++ #v[2]) 2

private def dummyBatchThree : BatchedTensor Nat #v[2] 3 :=
  Tensor.fill (#v[3] ++ #v[2]) 3

-- 同じ Layer 値に、異なる batch size の入力を順に渡せる。
private theorem dummyLayerAcceptsDifferentBatchSizes :
    let firstResult := dummyLayer.forward dummyBatchOne
    let secondResult := firstResult.2.forward dummyBatchThree
    firstResult.1.data.size = 2 ∧ secondResult.1.data.size = 6 := by
  decide

-- backward は parameter を変更せず、計算した gradient を既存値へ加算する。
private theorem dummyLayerAccumulatesParameterGradients :
    let firstForward := dummyForwardState dummyBatchOne {}
    let secondForward := dummyForwardState dummyBatchThree firstForward.2
    let firstBackward := dummyBackwardState secondForward.1 secondForward.2
    let secondBackward := dummyBackwardState firstForward.1 firstBackward.2
    secondBackward.2.cachedBatchSizes = [] ∧
      secondBackward.2.accumulatedGradient = 12 := by
  simp [dummyForwardState, dummyBackwardState, dummyBatchOne,
    dummyBatchThree, Tensor.fill, shapeSize]

end DL
