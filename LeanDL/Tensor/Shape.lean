import Init.Data.Vector.Lemmas
import LeanDL.Tensor.Basic

namespace DL.Tensor

/-- 指定された軸をshapeから取り除く。 -/
def eraseAxis {rank : Nat}
    (shape : Vector Nat rank) (axis : Fin rank) : Vector Nat (rank - 1) :=
  shape.eraseIdx axis.val axis.isLt


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

private theorem compatible_comm (left right : Nat) :
    compatible left right = compatible right left := by
  simp [compatible, Bool.beq_comm, Bool.or_comm, Bool.or_left_comm]

private theorem broadcastDim_comm_of_compatible (left right : Nat)
    (h : compatible left right) :
    broadcastDim left right = broadcastDim right left := by
  simp [compatible] at h
  unfold broadcastDim
  split <;> split <;> simp_all

/-- rankの順序を入れ替えたbroadcast結果を、元のrank順序へ移す。 -/
def castBroadcastResult {leftRank rightRank : Nat} :
    Option (Vector Nat (max rightRank leftRank)) →
      Option (Vector Nat (max leftRank rightRank))
  | none => none
  | some shape => some (Vector.cast (Nat.max_comm rightRank leftRank) shape)

private theorem getPadded_cast_target
    {rank target1 target2 : Nat}
    (shape : Vector Nat rank)
    (h1 : rank ≤ target1)
    (h2 : rank ≤ target2)
    (hTarget : target1 = target2)
    (i : Fin target1) :
    getPadded shape h1 i = getPadded shape h2 (Fin.cast hTarget i) := by
  subst target2
  rfl

/-- broadcastは左右のshapeを入れ替えても、rankのcastを除いて同じ結果になる。 -/
theorem broadcast_comm
    {leftRank rightRank : Nat}
    (left : Vector Nat leftRank)
    (right : Vector Nat rightRank) :
    broadcast left right = castBroadcastResult (broadcast right left) := by
  unfold broadcast
  let leftDim := getPadded left (Nat.le_max_left leftRank rightRank)
  let rightDim := getPadded right (Nat.le_max_right leftRank rightRank)
  let swappedRightDim := getPadded right (Nat.le_max_left rightRank leftRank)
  let swappedLeftDim := getPadded left (Nat.le_max_right rightRank leftRank)
  have hRightDim : ∀ i : Fin (max leftRank rightRank),
      rightDim i = swappedRightDim (Fin.cast (Nat.max_comm leftRank rightRank) i) := by
    intro i
    exact getPadded_cast_target right _ _ (Nat.max_comm leftRank rightRank) i
  have hLeftDim : ∀ i : Fin (max leftRank rightRank),
      leftDim i = swappedLeftDim (Fin.cast (Nat.max_comm leftRank rightRank) i) := by
    intro i
    exact getPadded_cast_target left _ _ (Nat.max_comm leftRank rightRank) i
  by_cases hCompatible : ∀ i, compatible (leftDim i) (rightDim i)
  · have hSwapped : ∀ i, compatible (swappedRightDim i) (swappedLeftDim i) := by
      intro i
      let j : Fin (max leftRank rightRank) :=
        Fin.cast (Nat.max_comm rightRank leftRank) i
      have h := hCompatible j
      have hr := hRightDim j
      have hl := hLeftDim j
      rw [compatible_comm] at h
      simpa [j, hr, hl] using h
    rw [if_pos hCompatible, if_pos hSwapped]
    simp only [castBroadcastResult, Option.some.injEq]
    ext i hi
    let j : Fin (max leftRank rightRank) := ⟨i, hi⟩
    have h := broadcastDim_comm_of_compatible
      (leftDim j) (rightDim j) (hCompatible j)
    have hr := hRightDim j
    have hl := hLeftDim j
    have hResult :
        broadcastDim (leftDim j) (rightDim j) =
          broadcastDim
            (swappedRightDim (Fin.cast (Nat.max_comm leftRank rightRank) j))
            (swappedLeftDim (Fin.cast (Nat.max_comm leftRank rightRank) j)) := by
      calc
        broadcastDim (leftDim j) (rightDim j) =
            broadcastDim (rightDim j) (leftDim j) := h
        _ = _ := by rw [hr, hl]
    simpa [j, leftDim, rightDim, swappedRightDim, swappedLeftDim, Vector.get] using hResult
  · have hSwapped : ¬ ∀ i, compatible (swappedRightDim i) (swappedLeftDim i) := by
      intro h
      apply hCompatible
      intro i
      have hs := h (Fin.cast (Nat.max_comm leftRank rightRank) i)
      have hr := hRightDim i
      have hl := hLeftDim i
      rw [compatible_comm] at hs
      simpa [hr, hl] using hs
    rw [if_neg hCompatible, if_neg hSwapped]
    rfl

/-- 同じrankのshapeを `broadcast` の結果rankへ移したもの。 -/
def broadcastSelfShape {rank : Nat}
    (shape : Vector Nat rank) : Vector Nat (max rank rank) :=
  Vector.cast (by simp) shape

/-- suffixをprefix付きshapeのrankへ移したbroadcast結果shape。 -/
def broadcastAppendShape
    {prefixRank suffixRank : Nat}
    (prefixShape : Vector Nat prefixRank)
    (suffixShape : Vector Nat suffixRank) :
    Vector Nat (max suffixRank (prefixRank + suffixRank)) :=
  Vector.cast (by omega) (prefixShape ++ suffixShape)

/--
短いshapeが長いshapeのsuffixと一致する場合、broadcast結果は長いshapeになる。
-/
theorem broadcast_suffix_append
    {prefixRank suffixRank : Nat}
    (prefixShape : Vector Nat prefixRank)
    (suffixShape : Vector Nat suffixRank) :
    broadcast suffixShape (prefixShape ++ suffixShape) =
      some (broadcastAppendShape prefixShape suffixShape) := by
  simp [broadcast, broadcastAppendShape, compatible, broadcastDim, getPadded]
  constructor
  · intro i
    by_cases h : i.val < prefixRank
    · simp [h]
    · have hiTotal : i.val < prefixRank + suffixRank := by omega
      have hiSuffix : i.val - prefixRank < suffixRank := by omega
      have hget :
          (prefixShape ++ suffixShape).get ⟨i.val, hiTotal⟩ =
            suffixShape.get ⟨i.val - prefixRank, hiSuffix⟩ := by
        change
          (prefixShape.toArray ++ suffixShape.toArray)[i.val]'(by simpa using hiTotal) =
            suffixShape.toArray[i.val - prefixRank]'(by simpa using hiSuffix)
        rw [Array.getElem_append_right (by simp; omega)]
        simp
      simp [h, hget]
  · ext i hi
    simp only [Vector.getElem_ofFn, Vector.getElem_cast]
    by_cases h : i < prefixRank
    · simp [h, Vector.get]
    · have hiTotal : i < prefixRank + suffixRank := by omega
      have hiSuffix : i - prefixRank < suffixRank := by omega
      have hget :
          (prefixShape ++ suffixShape).get ⟨i, hiTotal⟩ =
            suffixShape.get ⟨i - prefixRank, hiSuffix⟩ := by
        change
          (prefixShape.toArray ++ suffixShape.toArray)[i]'(by simpa using hiTotal) =
            suffixShape.toArray[i - prefixRank]'(by simpa using hiSuffix)
        rw [Array.getElem_append_right (by simp; omega)]
        simp
      by_cases hone : suffixShape.get ⟨i - prefixRank, hiSuffix⟩ = 1
      · rw [if_pos (by
          intro hp
          simpa [Vector.get] using hone)]
        rfl
      · simp [hone, h]
        simpa using hget.symm

private theorem broadcast_suffix_cast_append
    {prefixRank suffixRank resultRank : Nat}
    (prefixShape : Vector Nat prefixRank)
    (suffixShape : Vector Nat suffixRank)
    (hRank : prefixRank + suffixRank = resultRank) :
    broadcast suffixShape (Vector.cast hRank (prefixShape ++ suffixShape)) =
      some (Vector.cast (by omega) (prefixShape ++ suffixShape)) := by
  subst resultRank
  simpa [broadcastAppendShape] using
    (broadcast_suffix_append prefixShape suffixShape)

/--
shapeを自分自身とbroadcastすると、次元は変化しない。
これは `broadcast_suffix_append` のprefixが空である場合に相当する。
-/
theorem broadcast_self {rank : Nat} (shape : Vector Nat rank) :
    broadcast shape shape = some (broadcastSelfShape shape) := by
  have h := broadcast_suffix_cast_append
    (#v[] : Vector Nat 0) shape (resultRank := rank) (by omega)
  simpa [broadcastSelfShape, Vector.cast] using h

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
