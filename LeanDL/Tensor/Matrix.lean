import LeanDL.Tensor.Operations
import LeanDL.Tensor.Manipulation

namespace DL.Tensor

private abbrev emptyBatch : Vector Nat 0 := #v[]

private theorem shapeSize_empty_append
    {rank : Nat} (shape : Vector Nat rank) :
    shapeSize shape = shapeSize (emptyBatch ++ shape) := by
  unfold shapeSize
  rw [Vector.foldl_append]
  simp

/--
通常の2次元Tensor同士で行列積を計算する。

内部では空のbatch shapeを付けて `matmul` を呼び、結果から空のbatch shapeを外す。
reshapeは同じArrayを再利用するため、入力のコピーや並べ替えは行わない。
-/
def matmul2D
    {α β γ : Type}
    {rows inner1 inner2 cols : Nat}
    (transposeLeft transposeRight : Bool)
    (left : Tensor α (matrixShape transposeLeft rows inner1))
    (right : Tensor β (matrixShape transposeRight inner2 cols))
    (mul : α → β → γ)
    (sum : γ → γ → γ)
    (zero : γ)
    (hInnerEq : inner1 = inner2) :
    Tensor γ #v[rows, cols] :=
  let leftWithBatch :
      Tensor α (emptyBatch ++ matrixShape transposeLeft rows inner1) :=
    reshape left _ (shapeSize_empty_append _)
  let rightWithBatch :
      Tensor β (emptyBatch ++ matrixShape transposeRight inner2 cols) :=
    reshape right _ (shapeSize_empty_append _)
  let resultWithBatch := matmul
    (resultBatch := emptyBatch) transposeLeft transposeRight
    leftWithBatch rightWithBatch mul sum zero hInnerEq (by decide)
  reshape resultWithBatch #v[rows, cols] (shapeSize_empty_append _).symm

private theorem broadcast_matrix_vector (rows cols : Nat) :
    broadcast #v[rows, cols] #v[cols] = some #v[rows, cols] := by
  have h := broadcast_comm (#v[rows, cols]) (#v[cols])
  have hSuffix := broadcast_suffix_append (#v[rows]) (#v[cols])
  simp [broadcastAppendShape] at hSuffix
  rw [hSuffix] at h
  simpa [castBroadcastResult] using h

/--
`matrix[row, col]` と `vector[col]` に `f` を適用し、vectorを行方向へbroadcastする。
-/
def zipWithMatrixVector
    {α β γ : Type}
    {rows cols : Nat}
    (matrix : Tensor α #v[rows, cols])
    (vector : Tensor β #v[cols])
    (f : α → β → γ) : Tensor γ #v[rows, cols] :=
  zipWith matrix vector f (broadcast_matrix_vector rows cols)

/-- vectorを行列の列方向に合わせ、`f vector[col] matrix[row, col]` を計算する。 -/
def zipWithVectorMatrix
    {α β γ : Type}
    {rows cols : Nat}
    (vector : Tensor α #v[cols])
    (matrix : Tensor β #v[rows, cols])
    (f : α → β → γ) : Tensor γ #v[rows, cols] :=
  zipWithMatrixVector matrix vector fun matrixValue vectorValue =>
    f vectorValue matrixValue

-- ここから先は検証用の example と、それに付随する private 定義。

private def matmul2DExampleLeft : Tensor Nat #v[2, 3] where
  data := #[1, 2, 3, 4, 5, 6]
  hsize := by decide

private def matmul2DExampleRight : Tensor Nat #v[3, 2] where
  data := #[7, 8, 9, 10, 11, 12]
  hsize := by decide

private def matmul2DExampleRightTransposed : Tensor Nat #v[2, 3] where
  data := #[7, 9, 11, 8, 10, 12]
  hsize := by decide

example :
    (matmul2D false false matmul2DExampleLeft matmul2DExampleRight
      (· * ·) (· + ·) 0 (by rfl)).data = #[58, 64, 139, 154] := by
  decide

example :
    (matmul2D false true matmul2DExampleLeft matmul2DExampleRightTransposed
      (· * ·) (· + ·) 0 (by rfl)).data = #[58, 64, 139, 154] := by
  decide

private def matrixVectorExampleMatrix : Tensor Nat #v[2, 3] where
  data := #[1, 2, 3, 4, 5, 6]
  hsize := by decide

private def matrixVectorExampleVector : Tensor Nat #v[3] where
  data := #[10, 20, 30]
  hsize := by decide

example :
    (zipWithMatrixVector matrixVectorExampleMatrix matrixVectorExampleVector
      (· + ·)).data = #[11, 22, 33, 14, 25, 36] := by
  decide

end DL.Tensor
