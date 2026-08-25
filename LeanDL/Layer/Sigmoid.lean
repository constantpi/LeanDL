import LeanDL.Layer.Activation

namespace DL.Sigmoid

/-- sigmoidをFloat値へ適用する。 -/
def forwardElement (input : Float) : Float :=
  1 / (1 + Float.exp (-input))

/-- cacheされたsigmoid出力を使って入力勾配を計算する。 -/
def backwardElement (output outputGradient : Float) : Float :=
  outputGradient * output * (1 - output)

/-- sample shapeを変えず、要素ごとにsigmoidを適用するFloat専用Layerを作る。 -/
def new
    {rank : Nat} {sampleShape : Vector Nat rank} :
    Layer Float sampleShape sampleShape :=
  Activation.newFromOutput forwardElement backwardElement

-- ここから先は検証用の example と、それに付随する private 定義。

private def testLayer : Layer Float #v[2] #v[2] := new

private def testInput : BatchedTensor Float #v[2] 2 :=
  Tensor.fill (#v[2] ++ #v[2]) 0

example : (forwardElement 0).toBits = (0.5 : Float).toBits := by
  native_decide

example : (backwardElement 0.5 2).toBits = (0.5 : Float).toBits := by
  native_decide

private theorem forwardCachesInputBatch :
    (testLayer.forward testInput).2.cachedBatchSize = some 2 := by
  exact Layer.cachedBatchSize_forward testLayer testInput

end DL.Sigmoid
