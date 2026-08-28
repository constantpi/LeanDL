import LeanDL.Tensor.Basic

namespace DL

/--
同じshapeを持つparameterとgradientから、新しいparameterを計算する状態付きの更新規則。

SGD、Momentum、Adamなどで異なる内部状態は `State` に隠蔽される。Optimizer自身は
parameterを所有せず、`stepState` の入力として現在値とgradientを受け取る。
-/
structure Optimizer
    (α : Type)
    {rank : Nat}
    (shape : Vector Nat rank)
    (State : Type) where
  /-- 現在のOptimizer状態。 -/
  state : State
  /-- parameterとgradientを使い、parameterとOptimizer状態を1 step更新する。 -/
  stepState :
    Tensor α shape →
    Tensor α shape →
    StateM State (Tensor α shape)

namespace Optimizer

/--
Optimizerを1 step実行する。

結果の第1要素は更新後のparameter、第2要素は更新後の状態を持つOptimizerである。
-/
def step
    {α : Type}
    {rank : Nat}
    {shape : Vector Nat rank}
    {State : Type}
    (optimizer : Optimizer α shape State)
    (parameter gradient : Tensor α shape) :
    Tensor α shape × Optimizer α shape State :=
  let (nextParameter, nextState) :=
    optimizer.stepState parameter gradient optimizer.state
  (nextParameter, { optimizer with state := nextState })

end Optimizer
end DL
