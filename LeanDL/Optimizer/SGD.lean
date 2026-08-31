import LeanDL.Optimizer.Parameter

namespace DL.SGD

/--
学習率を受け取り、SGD Optimizerを作る。

各要素について `parameter - learningRate * gradient` を計算する。SGDは更新間で
保持する履歴を必要としないため、Optimizer stateは `Unit` である。
-/
def new
    {α : Type} [Sub α] [Mul α]
    {rank : Nat}
    {shape : Vector Nat rank}
    (learningRate : α) : Optimizer α shape Unit where
  state := ()
  stepState := fun parameter gradient state =>
    -- stateはUnitなので変更されない
    let nextParameter := Tensor.zipWithSame parameter gradient fun value grad =>
      value - learningRate * grad
    (nextParameter, state)

-- ここから先は検証用の example と、それに付随する private 定義。

private def testValue : Tensor Int #v[3] where
  data := #[10, 20, 30]
  hsize := by decide

private def testGradient : Tensor Int #v[3] where
  data := #[1, 2, -1]
  hsize := by decide

example :
    let optimizer : Optimizer Int #v[3] Unit := new 2
    let result := optimizer.step testValue testGradient
    result.1.data = #[8, 16, 32] := by
  native_decide

example :
    let optimizer : Optimizer Int #v[3] Unit := new 2
    let parameter := (Parameter.new testValue optimizer).accumulate testGradient
    let updated := parameter.step
    updated.value.data = #[8, 16, 32] ∧
      updated.accumulatedGradient.data = #[0, 0, 0] := by
  native_decide

end DL.SGD
