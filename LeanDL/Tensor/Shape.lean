import Init.Data.Vector.Lemmas
import LeanDL.Tensor.Basic

namespace DL.Tensor


/-- NumPy の規則で2つの次元がbroadcast可能か判定する。 -/
private def compatible (left right : Nat) : Bool :=
  left == right || left == 1 || right == 1

/--
互換な2次元をbroadcastした結果。
`0` と `1` の結果を `0` にするため、単純な `max` は使用しない。
-/
private def broadcastDim (left right : Nat) : Nat :=
  if left == 1 then right else left

/-- 左を1で埋められた状態でgetする-/
private def getPadded {rank target : Nat} (shape : Vector Nat rank)
    (h : rank ≤ target) (i : Fin target) : Nat :=
  if hindex : i.val < target - rank then
    1
  else
    shape.get ⟨i.val - (target - rank), by omega⟩

/-- padded前に0があることは元から0を含むことと同値-/
private theorem padded_shape_nonzero_iff {rank target : Nat} (shape : Vector Nat rank)
    (h : rank ≤ target) :
    (∀ i, getPadded shape h i ≠ 0) ↔ (∀ i, shape.get i ≠ 0) := by
  constructor
  · intro hpad i
    by_cases is_zero : shape.get i = 0
    .
      -- 元のshapeに0がある場合、padded後も0がある
      simp [getPadded] at hpad
      have hpadAt := hpad ⟨i.val + (target - rank), by omega⟩
      simp [show ¬(i.val + (target - rank) < target - rank) from by omega] at hpadAt
      contradiction
    . exact is_zero
  · intro hshape i
    unfold getPadded
    split
    . simp
    . apply hshape


/--
2つの shape を NumPy の規則でbroadcastする。

末尾の次元から比較し、それぞれの次元が等しいか、どちらかが `1` なら成功する。
成功時のrankは入力rankの最大値となり、互換でない次元があれば `none` を返す。
-/
def broadcast {leftRank rightRank : Nat}
    (left : Vector Nat leftRank) (right : Vector Nat rightRank) :
    Option (Vector Nat (max leftRank rightRank)) :=
  let resultRank := max leftRank rightRank
  let leftDim := getPadded left (Nat.le_max_left leftRank rightRank)
  let rightDim := getPadded right (Nat.le_max_right leftRank rightRank)
  if ∀ i : Fin resultRank, compatible (leftDim i) (rightDim i) then
    some <| Vector.ofFn fun i =>
      broadcastDim (leftDim i) (rightDim i)
  else
    none

private theorem broadcastDim_pos_left {left right : Nat} :
    broadcastDim left right ≠ 0 → left ≠ 0 := by
  unfold broadcastDim
  split <;> simp_all

private theorem broadcastDim_pos_right {left right : Nat}
    (hcompat : compatible left right) :
    broadcastDim left right ≠ 0 → right ≠ 0 := by
  simp [compatible] at hcompat
  unfold broadcastDim
  split
  .
    omega
  .
    simp_all
    omega

/-- broadcastDimが正であるための条件-/
private theorem broadcastDim_pos {left right : Nat}
    (hcompat : compatible left right) :
    broadcastDim left right ≠ 0 ↔ left ≠ 0 ∧ right ≠ 0 := by
  constructor
  · intro h
    exact ⟨broadcastDim_pos_left h, broadcastDim_pos_right hcompat h⟩
  . intro h
    simp [broadcastDim]
    split <;> simp_all

/-- broadcast結果がsomeのときにresultに要素が存在するための条件。 -/
theorem broadcast_some_pos {leftRank rightRank : Nat}
    (left : Vector Nat leftRank)
    (right : Vector Nat rightRank)
    (result : Vector Nat (max leftRank rightRank))
    (h_some : broadcast left right = some result) :
    ¬ shape_is_zero result ↔
      (¬ shape_is_zero left) ∧ (¬ shape_is_zero right) := by
  simp [DL.Tensor.shape_nonzero_iff]
  simp only [broadcast,Option.ite_none_right_eq_some, Option.some.injEq] at h_some
  obtain ⟨h_compat, h_some⟩ := h_some
  simp [← h_some]
  constructor <;> intro h
  .
    constructor <;> intro i
    ·
      by_contra hleft
      have padd_iff := padded_shape_nonzero_iff left (Nat.le_max_left leftRank rightRank)
      have exists_zero : ¬ ∀ i, left.get i ≠ 0 := by
        push_neg
        exact ⟨i, hleft⟩
      have exists_zero_padded : ¬ ∀ i, getPadded left (Nat.le_max_left leftRank rightRank) i ≠ 0 := by
        intro hpad
        exact exists_zero (padd_iff.mp hpad)
      push_neg at exists_zero_padded
      obtain ⟨j, hj⟩ := exists_zero_padded
      have := (broadcastDim_pos (h_compat j)).mp (h j)
      exact this.left hj
    .
      by_contra hright
      have padd_iff := padded_shape_nonzero_iff right (Nat.le_max_right leftRank rightRank)
      have exists_zero : ¬ ∀ i, right.get i ≠ 0 := by
        push_neg
        exact ⟨i, hright⟩
      have exists_zero_padded : ¬ ∀ i, getPadded right (Nat.le_max_right leftRank rightRank) i ≠ 0 := by
        intro hpad
        exact exists_zero (padd_iff.mp hpad)
      push_neg at exists_zero_padded
      obtain ⟨j, hj⟩ := exists_zero_padded
      have broadcast_j := h j
      have := (broadcastDim_pos (h_compat j)).mp (h j)
      exact this.right hj
  . intro i
    -- padded後にも0はない
    have hpad_left := (padded_shape_nonzero_iff left (Nat.le_max_left leftRank rightRank)).mpr h.left i
    have hpad_right := (padded_shape_nonzero_iff right (Nat.le_max_right leftRank rightRank)).mpr h.right i
    apply (broadcastDim_pos (h_compat i)).mpr
    exact ⟨hpad_left, hpad_right⟩

-- 基本的なNumPy broadcastの例
example : broadcast #v[3, 1] #v[1, 4] = some #v[3, 4] := by decide
example : broadcast #v[5, 1, 4] #v[3, 1] = some #v[5, 3, 4] := by decide
example : broadcast #v[2, 3] #v[3] = some #v[2, 3] := by decide
example : broadcast #v[2, 3] #v[2] = none := by decide
example : broadcast #v[0, 3] #v[1, 3] = some #v[0, 3] := by decide
example : broadcast #v[1, 3] #v[0, 3] = some #v[0, 3] := by decide
example : broadcast #v[] #v[2, 3] = some #v[2, 3] := by decide

end DL.Tensor
