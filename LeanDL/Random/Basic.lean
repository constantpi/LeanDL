import Batteries.Data.Random.MersenneTwister

namespace DL.Random

open Batteries.Random.MersenneTwister

/--
明示的に受け渡す疑似乱数生成器の状態。

`engine` はMT19937-64の状態、`cachedNormal` はBox--Muller法が一度に生成する
二つ目の標準正規乱数を保持する。外部の時刻やグローバル状態は参照しない。
-/
structure Generator where
  engine : Batteries.Random.MersenneTwister.State mt19937_64
  cachedNormal : Option Float := none

namespace Generator

/-- 必須の64 bit seedから、決定的な疑似乱数生成器を作る。 -/
def fromSeed (seed : UInt64) : Generator where
  engine := mt19937_64.init (BitVec.ofNat 64 seed.toNat)

end Generator

/-- Generatorを明示的な状態として受け渡す乱数計算。 -/
abbrev RandomM := StateM Generator

/-- 次の64 bit符号なし整数を生成し、Generatorを1 step進める。 -/
def nextUInt64 : RandomM UInt64 :=
  fun generator =>
    let (word, nextEngine) := generator.engine.next
    (word.toNat.toUInt64, { generator with engine := nextEngine })

/--
`0 ≤ value < 1` の一様乱数を生成する。

IEEE 754 binary64の仮数精度に合わせ、64 bit乱数の上位53 bitを使用する。
-/
def uniform01 : RandomM Float := do
  let word ← nextUInt64
  let significand := word.toNat / 2048
  pure (significand.toFloat / 9007199254740992)

/--
`lower` と `upper` の間の一様乱数を生成する。

有限の値について `lower < upper` を満たす引数を想定する。
-/
def uniform (lower upper : Float) : RandomM Float := do
  let unit ← uniform01
  pure (lower + (upper - lower) * unit)

/--
平均0、標準偏差1の正規乱数を生成する。

Box--Muller法を使用する。一度の計算で得られる二つ目の値はGenerator内に保存し、
次の `standardNormal` 呼び出しで乱数engineを進めずに返す。
-/
def standardNormal : RandomM Float :=
  fun generator =>
    match generator.cachedNormal with
    | some cached =>
        (cached, { generator with cachedNormal := none })
    | none =>
        let (unit1, generator) := uniform01 generator
        let (unit2, generator) := uniform01 generator
        -- uniform01は0を含むため、1 - unit1を使ってlogの入力を (0, 1] にする。
        let positiveUnit := 1 - unit1
        let radius := Float.sqrt (-2 * Float.log positiveUnit)
        let angle := 6.283185307179586 * unit2
        let first := radius * Float.cos angle
        let second := radius * Float.sin angle
        (first, { generator with cachedNormal := some second })

/-- 指定した平均と標準偏差を持つ正規乱数を生成する。 -/
def normal (mean standardDeviation : Float) : RandomM Float := do
  let standard ← standardNormal
  pure (mean + standardDeviation * standard)

-- ここから先は検証用のexample。

private def seed : UInt64 := 42

example :
    let first := nextUInt64 (Generator.fromSeed seed)
    let second := nextUInt64 (Generator.fromSeed seed)
    first.1 = second.1 := by
  native_decide

example :
    let first := nextUInt64 (Generator.fromSeed seed)
    let second := nextUInt64 first.2
    first.1 ≠ second.1 := by
  native_decide

example :
    let first := nextUInt64 (Generator.fromSeed 1)
    let second := nextUInt64 (Generator.fromSeed 2)
    first.1 ≠ second.1 := by
  native_decide

example :
    let result := uniform01 (Generator.fromSeed seed)
    0 ≤ result.1 ∧ result.1 < 1 := by
  native_decide

example :
    let result := uniform (-3) 5 (Generator.fromSeed seed)
    (-3 : Float) ≤ result.1 ∧ result.1 < 5 := by
  native_decide

example :
    let first := standardNormal (Generator.fromSeed seed)
    first.2.cachedNormal.isSome = true := by
  native_decide

example :
    let first := standardNormal (Generator.fromSeed seed)
    let second := standardNormal first.2
    second.2.cachedNormal.isNone = true := by
  native_decide

example :
    let first := normal 10 2 (Generator.fromSeed seed)
    let second := normal 10 2 (Generator.fromSeed seed)
    first.1.toBits = second.1.toBits := by
  native_decide

end DL.Random
