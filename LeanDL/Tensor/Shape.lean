import Init.Data.Vector.Lemmas

namespace DL.Shape

/--
shape の先頭を `1` で埋め、指定された rank に揃える。
Broadcast は末尾の次元同士を比較するため、短い shape の左側を埋める。
-/
private def padLeft {rank target : Nat} (shape : Vector Nat rank)
    (h : rank ≤ target) : Vector Nat target :=
  (Vector.replicate (target - rank) 1 ++ shape).cast (Nat.sub_add_cancel h)

/-- NumPy の規則で2つの次元がbroadcast可能か判定する。 -/
private def compatible (left right : Nat) : Bool :=
  left == right || left == 1 || right == 1

/--
互換な2次元をbroadcastした結果。
`0` と `1` の結果を `0` にするため、単純な `max` は使用しない。
-/
private def broadcastDim (left right : Nat) : Nat :=
  if left == 1 then right else left

/--
2つの shape を NumPy の規則でbroadcastする。

末尾の次元から比較し、それぞれの次元が等しいか、どちらかが `1` なら成功する。
成功時のrankは入力rankの最大値となり、互換でない次元があれば `none` を返す。
-/
def broadcast {leftRank rightRank : Nat}
    (left : Vector Nat leftRank) (right : Vector Nat rightRank) :
    Option (Vector Nat (max leftRank rightRank)) :=
  let paddedLeft := padLeft left (Nat.le_max_left leftRank rightRank)
  let paddedRight := padLeft right (Nat.le_max_right leftRank rightRank)
  let dimensions := paddedLeft.zip paddedRight
  if dimensions.all fun dim => compatible dim.1 dim.2 then
    some (dimensions.map fun dim => broadcastDim dim.1 dim.2)
  else
    none

-- 基本的なNumPy broadcastの例
example : broadcast #v[3, 1] #v[1, 4] = some #v[3, 4] := by decide
example : broadcast #v[5, 1, 4] #v[3, 1] = some #v[5, 3, 4] := by decide
example : broadcast #v[2, 3] #v[3] = some #v[2, 3] := by decide
example : broadcast #v[2, 3] #v[2] = none := by decide
example : broadcast #v[0, 3] #v[1, 3] = some #v[0, 3] := by decide
example : broadcast #v[1, 3] #v[0, 3] = some #v[0, 3] := by decide
example : broadcast #v[] #v[2, 3] = some #v[2, 3] := by decide

end DL.Shape
