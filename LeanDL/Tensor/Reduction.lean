import LeanDL.Tensor.Indexing

namespace DL.Tensor

/--
指定された軸を、`initial` から `fold` で畳み込む。

出力shapeから `axis` は削除される。軸上の要素は添字の小さい順に
`fold accumulator element` として処理される。対象軸の長さが0なら、各出力要素は
`initial` のままになる。
-/
def foldAxis
    {α : Type}
    {rank : Nat}
    {shape : Vector Nat rank}
    (tensor : Tensor α shape)
    (axis : Fin rank)
    (initial : α)
    (fold : α → α → α) : Tensor α (eraseAxis shape axis) :=
  let resultShape := eraseAxis shape axis
  let eraseAxisSize := shape.get axis
  let data := Array.ofFn fun flatIndex : Fin (shapeSize resultShape) =>
    let resultIndex := to_multi_index resultShape flatIndex flatIndex.isLt
    let axisValues := Array.ofFn fun axisIndex : Fin eraseAxisSize =>
      tensor[Internal.insertAxisIndex axis resultIndex axisIndex]
    axisValues.foldl fold initial
  have hsize : data.size = shapeSize resultShape := by simp [data]
  { data, hsize }

-- ここから先は検証用の example と、それに付随する private 定義。

private def foldAxisExample : Tensor Nat #v[2, 3] where
  data := #[1, 2, 3, 4, 5, 6]
  hsize := by decide

example :
    (foldAxis foldAxisExample (0 : Fin 2) 0 (· + ·)).data = #[5, 7, 9] := by
  native_decide

example :
    (foldAxis foldAxisExample (1 : Fin 2) 0 (· + ·)).data = #[6, 15] := by
  native_decide

example :
    (foldAxis foldAxisExample (1 : Fin 2) 0 max).data = #[3, 6] := by
  native_decide

private def foldEmptyAxisExample : Tensor Nat #v[2, 0, 3] where
  data := #[]
  hsize := by decide

example :
    (foldAxis foldEmptyAxisExample (1 : Fin 3) 7 (· + ·)).data =
      #[7, 7, 7, 7, 7, 7] := by
  native_decide

end DL.Tensor
