import LeanDL.Tensor.Indexing

namespace DL.Tensor

/-- batch shapeに行列の2次元を追加したshapeの要素数。 -/
private theorem foldl_append_matrix
    {batchRank : Nat} (batch : Vector Nat batchRank) (rows cols : Nat) :
    (batch ++ #v[rows, cols]).foldl (· * ·) 1 =
      batch.foldl (· * ·) 1 * rows * cols := by
  rw [Vector.foldl_append]
  simp

/-- transpose flagを適用した行列の物理shape。論理shapeは常に `rows × cols`。 -/
def matrixShape (transpose : Bool) (rows cols : Nat) : Vector Nat 2 :=
  if transpose then #v[cols, rows] else #v[rows, cols]

/-- 論理的な `(row, col)` を、transpose flagに応じた物理添字へ変換する。 -/
private def matrixIndex
    (transpose : Bool)
    {rows cols : Nat}
    (row : Fin rows) (col : Fin cols) :
    Index (matrixShape transpose rows cols) := by
  cases transpose with
  | false =>
      exact {
        values := #v[row.val, col.val]
        isValid := by
          simp [matrixShape, index_in_bounds, Fin.forall_fin_succ, Vector.get,
            row.isLt, col.isLt]
      }
  | true =>
      exact {
        values := #v[col.val, row.val]
        isValid := by
          simp [matrixShape, index_in_bounds, Fin.forall_fin_succ, Vector.get,
            row.isLt, col.isLt]
      }

/--
最後の2次元を行列、それ以前の次元をbatchとして一般化行列積を計算する。

論理的な行列shapeはそれぞれ `rows × inner1` と `inner2 × cols` であり、
`hInnerEq` が行列積の内側次元の一致を保証する。`transposeLeft` または
`transposeRight` が `true` の場合、対応する入力は最後の2次元を逆順にした物理
shapeで受け取り、Arrayを転置せず添字だけを入れ替えて参照する。
batch次元は `hBroadcast` に従ってNumPyと同じ規則でbroadcastされる。

各出力要素は `zero` から始め、`sum acc (mul leftValue rightValue)` の順に
`inner1` 次元を左から畳み込む。したがって `inner1 = 0` の場合は `zero` になる。
-/
def matmul
    {α β γ : Type}
    {leftBatchRank rightBatchRank : Nat}
    {leftBatch : Vector Nat leftBatchRank}
    {rightBatch : Vector Nat rightBatchRank}
    {resultBatch : Vector Nat (max leftBatchRank rightBatchRank)}
    {rows inner1 inner2 cols : Nat}
    (transposeLeft transposeRight : Bool)
    (left : Tensor α (leftBatch ++ matrixShape transposeLeft rows inner1))
    (right : Tensor β (rightBatch ++ matrixShape transposeRight inner2 cols))
    (mul : α → β → γ)
    (sum : γ → γ → γ)
    (zero : γ)
    (hInnerEq : inner1 = inner2)
    (hBroadcast : broadcast leftBatch rightBatch = some resultBatch) :
    Tensor γ (resultBatch ++ #v[rows, cols]) :=
  let resultShape := resultBatch ++ #v[rows, cols]
  let resultSize := shapeSize resultShape
  let data := Array.ofFn fun outputIndex : Fin resultSize =>
    have hResultPos : 0 < resultBatch.foldl (· * ·) 1 * rows * cols := by
      rw [← foldl_append_matrix]
      exact Nat.zero_lt_of_lt outputIndex.isLt
    have hResultBatchPos : 0 < resultBatch.foldl (· * ·) 1 := by
      exact Nat.pos_of_mul_pos_right (Nat.pos_of_mul_pos_right hResultPos)
    have hRowsPos : 0 < rows := by
      exact Nat.pos_of_mul_pos_left (Nat.pos_of_mul_pos_right hResultPos)
    have hColsPos : 0 < cols := by
      exact Nat.pos_of_mul_pos_left hResultPos
    have hResultBatchNonzero : ¬shape_is_zero resultBatch := by
      simp [shape_is_zero, shapeSize]
      exact Nat.ne_of_gt hResultBatchPos
    have hSourceBatchNonzero :=
      (broadcast_some_pos leftBatch rightBatch resultBatch hBroadcast).mp
        hResultBatchNonzero
    let resultIndex := to_multi_index resultShape outputIndex outputIndex.isLt
    let resultBatchIndex : Index resultBatch :=
      Internal.prefixIndex (rightShape := #v[rows, cols]) resultIndex
    let leftBatchIndex := Internal.broadcastIndex resultBatch leftBatch
      (Nat.le_max_left leftBatchRank rightBatchRank)
      hSourceBatchNonzero.1 resultBatchIndex
    let rightBatchIndex := Internal.broadcastIndex resultBatch rightBatch
      (Nat.le_max_right leftBatchRank rightBatchRank)
      hSourceBatchNonzero.2 resultBatchIndex
    let row := (outputIndex.val / cols) % rows
    let col := outputIndex.val % cols
    let products := Array.ofFn fun innerIndex : Fin inner1 =>
      let rightInnerIndex : Fin inner2 := Fin.cast hInnerEq innerIndex
      let rowIndex : Fin rows := ⟨row, Nat.mod_lt _ hRowsPos⟩
      let colIndex : Fin cols := ⟨col, Nat.mod_lt _ hColsPos⟩
      let leftMatrixIndex := matrixIndex transposeLeft rowIndex innerIndex
      let rightMatrixIndex :=
        matrixIndex transposeRight rightInnerIndex colIndex
      let leftIndex := Internal.appendIndex leftBatchIndex leftMatrixIndex
      let rightIndex := Internal.appendIndex rightBatchIndex rightMatrixIndex
      mul left[leftIndex] right[rightIndex]
    products.foldl sum zero
  have hsize : data.size = resultSize := by simp [data]
  { data := data, hsize := hsize }

-- ここから先は検証用の example と、それに付随する private 定義。

private def matmulExampleLeft :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[2, 3]) where
  data := #[1, 2, 3, 4, 5, 6]
  hsize := by decide

private def matmulExampleRight :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[3, 2]) where
  data := #[7, 8, 9, 10, 11, 12]
  hsize := by decide

example :
    (matmul (resultBatch := (#v[] : Vector Nat 0)) false false
      matmulExampleLeft matmulExampleRight (· * ·) (· + ·) 0
      (by rfl) (by decide)).data =
      #[58, 64, 139, 154] := by
  decide

private def transposedMatmulExampleLeft :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[3, 2]) where
  data := #[1, 4, 2, 5, 3, 6]
  hsize := by decide

private def transposedMatmulExampleRight :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[2, 3]) where
  data := #[7, 9, 11, 8, 10, 12]
  hsize := by decide

-- 左入力だけを転置扱いにしても、Arrayを並べ替えず論理行列積を計算できる。
example :
    (matmul (resultBatch := (#v[] : Vector Nat 0)) true false
      transposedMatmulExampleLeft matmulExampleRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[58, 64, 139, 154] := by
  decide

-- 右入力だけを転置扱いにできる。
example :
    (matmul (resultBatch := (#v[] : Vector Nat 0)) false true
      matmulExampleLeft transposedMatmulExampleRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[58, 64, 139, 154] := by
  decide

-- 両入力のtranspose flagは独立に組み合わせられる。
example :
    (matmul (resultBatch := (#v[] : Vector Nat 0)) true true
      transposedMatmulExampleLeft transposedMatmulExampleRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[58, 64, 139, 154] := by
  decide

private def batchedMatmulExampleLeft : Tensor Nat (#v[2] ++ #v[2, 3]) where
  data := #[1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 2]
  hsize := by decide

private def batchedMatmulExampleRight : Tensor Nat (#v[1] ++ #v[3, 2]) where
  data := #[7, 8, 9, 10, 11, 12]
  hsize := by decide

example :
    (matmul (resultBatch := #v[2]) false false
      batchedMatmulExampleLeft batchedMatmulExampleRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[58, 64, 139, 154, 25, 28, 56, 62] := by
  decide

private def transposedBatchedMatmulExampleLeft :
    Tensor Nat (#v[2] ++ #v[3, 2]) where
  data := #[1, 4, 2, 5, 3, 6, 2, 1, 0, 3, 1, 2]
  hsize := by decide

-- transpose flagを使っても、先頭のbatch broadcast規則は変わらない。
example :
    (matmul (resultBatch := #v[2]) true false
      transposedBatchedMatmulExampleLeft batchedMatmulExampleRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[58, 64, 139, 154, 25, 28, 56, 62] := by
  decide

private def unequalBatchRankLeft : Tensor Nat (#v[2, 1] ++ #v[1, 2]) where
  data := #[1, 2, 3, 4]
  hsize := by decide

private def unequalBatchRankRight : Tensor Nat (#v[3] ++ #v[2, 1]) where
  data := #[10, 20, 30, 40, 50, 60]
  hsize := by decide

example :
    (matmul (resultBatch := #v[2, 3]) false false
      unequalBatchRankLeft unequalBatchRankRight
      (· * ·) (· + ·) 0 (by rfl) (by decide)).data =
      #[50, 110, 170, 110, 250, 390] := by
  decide

private def emptyInnerMatmulLeft :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[2, 0]) where
  data := #[]
  hsize := by decide

private def emptyInnerMatmulRight :
    Tensor Nat ((#v[] : Vector Nat 0) ++ #v[0, 3]) where
  data := #[]
  hsize := by decide

example :
    (matmul (resultBatch := (#v[] : Vector Nat 0)) false false
      emptyInnerMatmulLeft emptyInnerMatmulRight
      (· * ·) (· + ·) 42 (by rfl) (by decide)).data =
      #[42, 42, 42, 42, 42, 42] := by
  decide

end DL.Tensor
