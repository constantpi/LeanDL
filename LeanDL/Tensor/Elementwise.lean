import LeanDL.Tensor.Indexing

namespace DL.Tensor

/--
2つのTensorをbroadcastしながら要素ごとに関数 `f` を適用する。

`hBroadcast` により、入力shapeがbroadcast可能であり、その結果が正確に
`resultShape` になることをコンパイル時に保証する。
-/
def zipWith
    {α β γ : Type}
    {leftRank rightRank : Nat}
    {leftShape : Vector Nat leftRank}
    {rightShape : Vector Nat rightRank}
    {resultShape : Vector Nat (max leftRank rightRank)}
    (left : Tensor α leftShape)
    (right : Tensor β rightShape)
    (f : α → β → γ)
    (hBroadcast : broadcast leftShape rightShape = some resultShape) :
    Tensor γ resultShape :=
  let resultSize := shapeSize resultShape
  let data := Array.ofFn fun index : Fin resultSize =>
    have result_size_pos : 0 < shapeSize resultShape := by
      subst resultSize
      have index_lt : index.val < shapeSize resultShape := index.isLt
      exact Nat.zero_lt_of_lt index_lt
    have result_nonzero : ¬shape_is_zero resultShape := by
      simp [shape_is_zero]
      omega
    have hBroadcast_some :=
      (broadcast_some_pos leftShape rightShape resultShape hBroadcast).mp
        result_nonzero

    let multiIndex := to_multi_index resultShape index index.isLt
    let leftIndex := Internal.broadcastIndex resultShape leftShape
      (Nat.le_max_left leftRank rightRank) hBroadcast_some.1 multiIndex
    let rightIndex := Internal.broadcastIndex resultShape rightShape
      (Nat.le_max_right leftRank rightRank) hBroadcast_some.2 multiIndex
    f (left[leftIndex]) (right[rightIndex])

  have hsize : data.size = resultSize := by simp [data]
  { data := data, hsize := hsize }

/-- 同じshapeの2つのTensorへ、要素ごとに関数 `f` を適用する。 -/
def zipWithSame
    {α β γ : Type}
    {rank : Nat}
    {shape : Vector Nat rank}
    (left : Tensor α shape)
    (right : Tensor β shape)
    (f : α → β → γ) : Tensor γ shape :=
  let result := zipWith left right f (broadcast_self shape)
  have hShapeSize : shapeSize (broadcastSelfShape shape) = shapeSize shape := by
    simp [shapeSize, broadcastSelfShape, Vector.foldl, Vector.cast]
  {
    data := result.data
    hsize := result.hsize.trans hShapeSize
  }

-- ここから先は検証用の example と、それに付随する private 定義。

private def broadcastExampleLeft : Tensor Nat #v[2, 1] where
  data := #[10, 20]
  hsize := by decide

private def broadcastExampleRight : Tensor Nat #v[3] where
  data := #[1, 2, 3]
  hsize := by decide

private def broadcastExampleResult : Tensor Nat #v[2, 3] :=
  zipWith broadcastExampleLeft broadcastExampleRight (· + ·) (by decide)

example : broadcastExampleResult.data = #[11, 12, 13, 21, 22, 23] := by decide

private def scalarExample : Tensor Nat #v[] where
  data := #[10]
  hsize := by decide

private def vectorExample : Tensor Nat #v[2] where
  data := #[1, 2]
  hsize := by decide

example :
    (zipWith (resultShape := #v[2]) scalarExample vectorExample (· + ·)
      (by decide)).data = #[11, 12] := by
  decide

private def emptyExample : Tensor Nat #v[0, 1] where
  data := #[]
  hsize := by decide

example :
    (zipWith (resultShape := #v[0, 3]) emptyExample broadcastExampleRight
      (· + ·) (by decide)).data = #[] := by
  decide

end DL.Tensor
