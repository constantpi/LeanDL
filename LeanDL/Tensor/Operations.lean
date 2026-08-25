import LeanDL.Tensor.Shape
import LeanDL.Tensor.Basic

namespace DL.Tensor

/-- Index resultShapeからIndex shapeへの変換 -/
private def broadcastIndex {rankResult rankSource : Nat}
    (resultShape : Vector Nat rankResult)
    (sourceShape : Vector Nat rankSource)
    (h_rank : rankSource ≤ rankResult)
    (h_nonzero : ¬shape_is_zero sourceShape)
    (index : Index resultShape) : Index sourceShape :=
  let sourceIndexValues : Vector Nat rankSource := Vector.ofFn fun i =>
    let sourceDim := sourceShape.get i
    let i' : Fin rankResult := ⟨i.val + (rankResult - rankSource), by omega⟩
    let index_i := index.values.get i'
    if index_i < sourceDim then index_i else 0

  have isValid : index_in_bounds sourceShape sourceIndexValues := by
    simp [index_in_bounds, sourceIndexValues]
    intro i
    let i' : Fin rankResult := ⟨i.val + (rankResult - rankSource), by omega⟩
    split
    .
      rename_i hindex
      exact hindex
    .
      rw [shape_nonzero_iff] at h_nonzero
      have := h_nonzero i
      omega

  {
    values := sourceIndexValues
    isValid := isValid
  }


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
    have hBroadcast_some := (broadcast_some_pos leftShape rightShape resultShape hBroadcast).mp result_nonzero

    let multiIndex := to_multi_index resultShape index index.isLt
    let leftIndex := broadcastIndex resultShape leftShape (Nat.le_max_left leftRank rightRank) hBroadcast_some.1 multiIndex
    let rightIndex := broadcastIndex resultShape rightShape (Nat.le_max_right leftRank rightRank) hBroadcast_some.2 multiIndex
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

/--
削除済みの軸を添字へ戻す。

`axisIndex` が削除した軸の値で、それ以外の値は `index` から引き継ぐ。
-/
private def insertAxisIndex
    {rank : Nat}
    {shape : Vector Nat rank}
    (axis : Fin rank)
    (index : Index (eraseAxis shape axis))
    (axisIndex : Fin (shape.get axis)) : Index shape :=
  let values : Vector Nat rank := Vector.ofFn fun i =>
    if hBefore : i.val < axis.val then
      index.values.get ⟨i.val, by omega⟩
    else if hAxis : i.val = axis.val then
      axisIndex.val
    else
      index.values.get ⟨i.val - 1, by omega⟩
  have isValid : index_in_bounds shape values := by
    intro i
    change values[i.val] < shape[i.val]
    simp only [values, Vector.getElem_ofFn]
    split
    · rename_i hBefore
      have h := index.isValid ⟨i.val, by omega⟩
      change index.values[i.val] <
        (shape.eraseIdx axis.val axis.isLt)[i.val] at h
      rw [Vector.getElem_eraseIdx_of_lt axis.isLt (by omega) hBefore] at h
      simpa [Vector.get] using h
    · rename_i hNotBefore
      split
      · rename_i hAxis
        have hi : i = axis := Fin.ext hAxis
        subst i
        exact axisIndex.isLt
      · rename_i hNotAxis
        have hAfter : axis.val < i.val := by omega
        have h := index.isValid ⟨i.val - 1, by omega⟩
        have hIndex : i.val - 1 + 1 = i.val := by omega
        change index.values[i.val - 1] <
          (shape.eraseIdx axis.val axis.isLt)[i.val - 1] at h
        rw [Vector.getElem_eraseIdx_of_ge axis.isLt (by omega) (by omega)] at h
        simpa [Vector.get, hIndex] using h
  { values, isValid }

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
  let erase_ax := shape.get axis
  let data := Array.ofFn fun flatIndex : Fin (shapeSize resultShape) =>
    let resultIndex := to_multi_index resultShape flatIndex flatIndex.isLt
    let axisValues := Array.ofFn fun axisIndex : Fin erase_ax =>
      tensor[insertAxisIndex axis resultIndex axisIndex]
    axisValues.foldl fold initial
  have hsize : data.size = shapeSize resultShape := by simp [data]
  { data, hsize }

/-- batch shape に行列の2次元を追加した shape の要素数。 -/
private theorem foldl_append_matrix
    {batchRank : Nat} (batch : Vector Nat batchRank) (rows cols : Nat) :
    (batch ++ #v[rows, cols]).foldl (· * ·) 1 =
      batch.foldl (· * ·) 1 * rows * cols := by
  rw [Vector.foldl_append]
  simp

/-- 2つの安全な添字を、連結した shape の安全な添字へ結合する。 -/
private def appendIndex
    {leftRank rightRank : Nat}
    {leftShape : Vector Nat leftRank}
    {rightShape : Vector Nat rightRank}
    (leftIndex : Index leftShape)
    (rightIndex : Index rightShape) :
    Index (leftShape ++ rightShape) where
  values := leftIndex.values ++ rightIndex.values
  isValid := by
    intro i
    change
      (leftIndex.values ++ rightIndex.values)[i.val] <
        (leftShape ++ rightShape)[i.val]
    rw [Vector.getElem_append, Vector.getElem_append]
    split
    · exact leftIndex.isValid ⟨i.val, by omega⟩
    · exact rightIndex.isValid ⟨i.val - leftRank, by omega⟩

/-- 連結 shape の安全な添字から、先頭側の添字を取り出す。 -/
private def prefixIndex
    {leftRank rightRank : Nat}
    {leftShape : Vector Nat leftRank}
    {rightShape : Vector Nat rightRank}
    (index : Index (leftShape ++ rightShape)) : Index leftShape where
  values := Vector.ofFn fun i => index.values.get ⟨i.val, by omega⟩
  isValid := by
    intro i
    have h := index.isValid ⟨i.val, by omega⟩
    simpa [Vector.get, Vector.getElem_append] using h

/-- transpose flagを適用した行列の物理 shape。論理 shape は常に `rows × cols`。 -/
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
最後の2次元を行列、それ以前の次元を batch として一般化行列積を計算する。

論理的な行列 shape はそれぞれ `rows × inner1` と `inner2 × cols` であり、
`hInnerEq` が行列積の内側次元の一致を保証する。`transposeLeft` または
`transposeRight` が `true` の場合、対応する入力は最後の2次元を逆順にした物理
shapeで受け取り、Arrayを転置せず添字だけを入れ替えて参照する。
batch 次元は `hBroadcast` に従って NumPy と同じ規則で broadcast される。

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
      prefixIndex (rightShape := #v[rows, cols]) resultIndex
    let leftBatchIndex := broadcastIndex resultBatch leftBatch
      (Nat.le_max_left leftBatchRank rightBatchRank)
      hSourceBatchNonzero.1 resultBatchIndex
    let rightBatchIndex := broadcastIndex resultBatch rightBatch
      (Nat.le_max_right leftBatchRank rightBatchRank)
      hSourceBatchNonzero.2 resultBatchIndex
    let row := (outputIndex.val / cols) % rows
    let col := outputIndex.val % cols
    let products := Array.ofFn fun innerIndex : Fin inner1 =>
      let rightInnerIndex : Fin inner2 := Fin.cast hInnerEq innerIndex
      let rowIndex : Fin rows := ⟨row, Nat.mod_lt _ hRowsPos⟩
      let colIndex : Fin cols := ⟨col, Nat.mod_lt _ hColsPos⟩
      let leftMatrixIndex :=
        matrixIndex transposeLeft rowIndex innerIndex
      let rightMatrixIndex :=
        matrixIndex transposeRight rightInnerIndex colIndex
      let leftIndex := appendIndex leftBatchIndex leftMatrixIndex
      let rightIndex := appendIndex rightBatchIndex rightMatrixIndex
      mul left[leftIndex] right[rightIndex]
    products.foldl sum zero
  have hsize : data.size = resultSize := by simp [data]
  { data := data, hsize := hsize }

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
    (zipWith (resultShape := #v[2]) scalarExample vectorExample (· + ·) (by decide)).data = #[11, 12] := by
  decide

private def emptyExample : Tensor Nat #v[0, 1] where
  data := #[]
  hsize := by decide

example :
    (zipWith (resultShape := #v[0, 3]) emptyExample broadcastExampleRight (· + ·) (by decide)).data = #[] := by
  decide

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
