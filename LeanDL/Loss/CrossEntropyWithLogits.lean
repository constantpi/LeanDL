import LeanDL.Layer.Basic
import LeanDL.Tensor.Matrix
import LeanDL.Tensor.Reduction
import LeanDL.Tensor.Tactics

namespace DL.Loss.CrossEntropyWithLogits

/-- batch内の損失を加算するか、batch sizeで割って平均するかを指定する。 -/
inductive Reduction where
  | sum
  | mean
  deriving DecidableEq

/-- Cross Entropyの損失値と、入力logitsに対する勾配。 -/
structure Result (batchSize classes : Nat) where
  value : Float
  gradient : BatchedTensor Float #v[classes] batchSize

private def batchedVectorAsMatrix
    {α : Type} {batchSize features : Nat}
    (tensor : BatchedTensor α #v[features] batchSize) :
    Tensor α #v[batchSize, features] :=
  Tensor.reshape tensor _ (by tensor_shape)

private def matrixAsBatchedVector
    {α : Type} {batchSize features : Nat}
    (tensor : Tensor α #v[batchSize, features]) :
    BatchedTensor α #v[features] batchSize :=
  Tensor.reshape tensor _ (by tensor_shape)

private def vectorAsColumn
    {α : Type} {rows : Nat}
    (tensor : Tensor α #v[rows]) : Tensor α #v[rows, 1] :=
  Tensor.reshape tensor _ (by tensor_shape)

private def negativeInfinity : Float :=
  -1 / 0

private def maximum (left right : Float) : Float :=
  if left < right then right else left

private def reductionScale
    (reduction : Reduction) (batchSize : Nat) : Float :=
  match reduction with
  | .sum => 1
  | .mean =>
      if batchSize = 0 then 0 else 1 / batchSize.toFloat

/-- クラス番号形式のラベルをone-hot Tensorへ変換する。 -/
private def oneHot
    {batchSize classes : Nat}
    (labels : Vector (Fin classes) batchSize) :
    Tensor Float #v[batchSize, classes] :=
  Tensor.ofFn #v[batchSize, classes] fun index =>
    -- let [batchValue, classValue] := index.values
    let batchValue := index.values[0]
    let classValue := index.values[1]
    have hBatch : batchValue < batchSize := by
      have h := index.isValid (0 : Fin 2)
      simpa [Vector.get] using h
    have hClass : classValue < classes := by
      have h := index.isValid (1 : Fin 2)
      simpa [Vector.get] using h
    let batchIndex : Fin batchSize := ⟨batchValue, hBatch⟩
    let classIndex : Fin classes := ⟨classValue, hClass⟩
    if classIndex = labels.get batchIndex then 1 else 0

/--
logitsと正解クラスから、数値的に安定したCross Entropyとlogits勾配を計算する。

モデルの出力にはSoftmaxを適用せず、その直前の生の値を `logits` として渡す。
損失は最大値を引いたlogitsに対するlog-sum-expから計算し、勾配は
`softmax(logits) - oneHot(labels)` とする。`.mean` の場合は損失と勾配の両方を
batch sizeで割る。空batchのmeanは損失0、空の勾配として扱う。
-/
def forward
    {batchSize classes : Nat}
    (logits : BatchedTensor Float #v[classes] batchSize)
    (labels : Vector (Fin classes) batchSize)
    (reduction : Reduction := .mean) : Result batchSize classes :=
  let logitsMatrix := batchedVectorAsMatrix logits
  let rowMaximums := Tensor.foldAxis logitsMatrix (1 : Fin 2) negativeInfinity maximum
  let centeredLogits := Tensor.zipWithMatrixColumn
    logitsMatrix (vectorAsColumn rowMaximums) (· - ·)
  let exponentials := Tensor.map centeredLogits Float.exp
  let rowExponentialSums := Tensor.foldAxis
    exponentials (1 : Fin 2) 0 (· + ·)
  let logRowExponentialSums := Tensor.map rowExponentialSums Float.log
  let logProbabilities := Tensor.zipWithMatrixColumn
    centeredLogits (vectorAsColumn logRowExponentialSums) (· - ·)
  let targets := oneHot labels
  let elementLosses := Tensor.zipWithSame
    logProbabilities targets fun logProbability target =>
      -(target * logProbability)
  let sampleLosses := Tensor.foldAxis elementLosses (1 : Fin 2) 0 (· + ·)
  let totalLoss := Tensor.foldAxis sampleLosses (0 : Fin 1) 0 (· + ·)
  let scale := reductionScale reduction batchSize
  let value := Tensor.get totalLoss (tensor_index% #v[]) * scale
  let probabilities := Tensor.zipWithMatrixColumn
    exponentials (vectorAsColumn rowExponentialSums) (· / ·)
  let gradientMatrix := Tensor.map
    (Tensor.zipWithSame probabilities targets (· - ·)) (· * scale)
  {
    value
    gradient := matrixAsBatchedVector gradientMatrix
  }

-- ここから先は検証用のexample。

private def equalLogits : BatchedTensor Float #v[2] 2 where
  data := #[0, 0, 0, 0]
  hsize := by decide

private def equalLabels : Vector (Fin 2) 2 :=
  #v[0, 1]

example :
    let result := forward equalLogits equalLabels
    result.value.toBits = (Float.log 2).toBits ∧
      result.gradient.data.map Float.toBits =
        #[(-0.25 : Float), 0.25, 0.25, -0.25].map Float.toBits := by
  native_decide

example :
    let result := forward equalLogits equalLabels .sum
    result.value.toBits = (Float.log 2 + Float.log 2).toBits ∧
      result.gradient.data.map Float.toBits =
        #[(-0.5 : Float), 0.5, 0.5, -0.5].map Float.toBits := by
  native_decide

-- 大きなlogitでもexpを取る前に最大値を引くためoverflowしない。
private def largeLogits : BatchedTensor Float #v[2] 1 where
  data := #[1000, 1000]
  hsize := by decide

example :
    let result := forward largeLogits (#v[0] : Vector (Fin 2) 1)
    result.value.toBits = (Float.log 2).toBits ∧
      result.gradient.data.map Float.toBits =
        #[(-0.5 : Float), 0.5].map Float.toBits := by
  native_decide

-- 正解classのlogitが最大値から大きく離れていても、log(softmax)を経由しない。
private def separatedLogits : BatchedTensor Float #v[2] 1 where
  data := #[1000, 0]
  hsize := by decide

example :
    let result := forward separatedLogits (#v[1] : Vector (Fin 2) 1)
    result.value.toBits = (1000 : Float).toBits ∧
      result.gradient.data.map Float.toBits =
        #[(1 : Float), -1].map Float.toBits := by
  native_decide

example :
    let logits : BatchedTensor Float #v[3] 0 := Tensor.fill _ 0
    let labels : Vector (Fin 3) 0 := #v[]
    let result := forward logits labels
    result.value.toBits = (0 : Float).toBits ∧
      result.gradient.data.map Float.toBits = #[] := by
  native_decide

end DL.Loss.CrossEntropyWithLogits
