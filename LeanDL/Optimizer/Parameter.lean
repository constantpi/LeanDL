import LeanDL.Optimizer.Basic
import LeanDL.Tensor.Elementwise

namespace DL

/--
学習可能なTensor parameter。

現在値、蓄積済みgradient、そのparameter専用のOptimizerをひとまとめにする。
-/
structure Parameter
    (α : Type)
    {rank : Nat}
    (shape : Vector Nat rank)
    (OptimizerState : Type) where
  value : Tensor α shape
  accumulatedGradient : Tensor α shape
  optimizer : Optimizer α shape OptimizerState

namespace Parameter

/-- gradientをzeroで初期化したParameterを作る。 -/
def new
    {α : Type} [OfNat α 0]
    {rank : Nat}
    {shape : Vector Nat rank}
    {OptimizerState : Type}
    (value : Tensor α shape)
    (optimizer : Optimizer α shape OptimizerState) :
    Parameter α shape OptimizerState := {
  value
  accumulatedGradient := Tensor.fill shape 0
  optimizer
}

/-- 新しいgradientを既存のgradientへ要素ごとに加算する。 -/
def accumulate
    {α : Type} [Add α]
    {rank : Nat}
    {shape : Vector Nat rank}
    {OptimizerState : Type}
    (parameter : Parameter α shape OptimizerState)
    (gradient : Tensor α shape) : Parameter α shape OptimizerState :=
  { parameter with
    accumulatedGradient := Tensor.zipWithSame
      parameter.accumulatedGradient gradient (· + ·) }

/-- parameter値を変えず、蓄積済みgradientをzeroへ戻す。 -/
def zeroGrad
    {α : Type} [OfNat α 0]
    {rank : Nat}
    {shape : Vector Nat rank}
    {OptimizerState : Type}
    (parameter : Parameter α shape OptimizerState) :
    Parameter α shape OptimizerState :=
  { parameter with accumulatedGradient := Tensor.fill shape 0 }

/--
蓄積済みgradientでOptimizerを1 step実行し、gradientを自動的にzeroへ戻す。
-/
def step
    {α : Type} [OfNat α 0]
    {rank : Nat}
    {shape : Vector Nat rank}
    {OptimizerState : Type}
    (parameter : Parameter α shape OptimizerState) :
    Parameter α shape OptimizerState :=
  let (nextValue, nextOptimizer) := parameter.optimizer.step
    parameter.value parameter.accumulatedGradient
  {
    value := nextValue
    accumulatedGradient := Tensor.fill shape 0
    optimizer := nextOptimizer
  }

-- ここから先は検証用の example と、それに付随する private 定義。

private def addGradientOptimizer : Optimizer Int #v[2] Nat where
  state := 0
  stepState := fun value gradient stepCount =>
    (Tensor.zipWithSame value gradient (· + ·), stepCount + 1)

private def testValue : Tensor Int #v[2] where
  data := #[10, 20]
  hsize := by decide

private def testGradient : Tensor Int #v[2] where
  data := #[1, 2]
  hsize := by decide

example :
    let parameter := (new testValue addGradientOptimizer).accumulate testGradient
    let updated := parameter.step
    updated.value.data = #[11, 22] ∧
      updated.accumulatedGradient.data = #[0, 0] := by
  native_decide

example :
    let parameter := (new testValue addGradientOptimizer).accumulate testGradient
    parameter.zeroGrad.accumulatedGradient.data = #[0, 0] := by
  native_decide

end Parameter
end DL
