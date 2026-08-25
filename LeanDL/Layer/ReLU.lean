import LeanDL.Layer.Activation

namespace DL.ReLU

/-- ReLUを要素ごとに適用する。 -/
def forwardElement
    {α : Type} [LT α] [DecidableLT α] [OfNat α 0]
    (input : α) : α :=
  if 0 < input then input else 0

/-- cacheされたReLU出力を使って入力勾配を計算する。0での微分は0とする。 -/
def backwardElement
    {α : Type} [LT α] [DecidableLT α] [OfNat α 0]
    (output outputGradient : α) : α :=
  if 0 < output then outputGradient else 0

/-- sample shapeを変えず、要素ごとにReLUを適用するLayerを作る。 -/
def new
    {α : Type} [LT α] [DecidableLT α] [OfNat α 0]
    {rank : Nat} {sampleShape : Vector Nat rank} :
    Layer α sampleShape sampleShape :=
  Activation.newFromOutput forwardElement backwardElement

-- ここから先は検証用の example と、それに付随する private 定義。

private def testInput : BatchedTensor Int #v[3] 2 where
  data := #[-2, 0, 3, 4, -5, 0]
  hsize := by decide

private def testOutputGradient : BatchedTensor Int #v[3] 2 where
  data := #[-1, 2, -3, 4, -5, 6]
  hsize := by decide

private theorem forwardAndBackwardValues :
    let layer : Layer Int #v[3] #v[3] := new
    let forwardResult := layer.forward testInput
    let backwardResult := forwardResult.2.backward testOutputGradient
      (Layer.cachedBatchSize_forward layer testInput)
    forwardResult.1.data = #[0, 0, 3, 4, 0, 0] ∧
      backwardResult.1.data = #[0, 0, -3, 4, 0, 0] := by
  native_decide

end DL.ReLU
