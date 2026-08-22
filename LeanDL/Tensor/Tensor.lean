import Init.Data.Vector.Lemmas
import LeanDL.Tensor.Prod

namespace DL

structure Tensor (α : Type) {rank : Nat} (shape : Vector Nat rank) where
  data : Array α
  hsize : data.size = shape.foldl (· * ·) 1

namespace Tensor

def fill {α : Type} {rank : Nat} (shape : Vector Nat rank) (value : α) : Tensor α shape :=
  let size := shape.foldl (· * ·) 1
  let data := Array.replicate size value
  have hsize : data.size = size := by simp [data]
  { data := data, hsize := hsize }

def shape {α : Type} {rank : Nat} {shape : Vector Nat rank} (_t : Tensor α shape) : Vector Nat rank :=
  shape

/-- 指定された個数の空白を作る。 -/
private def spaces (count : Nat) : String :=
  (List.replicate count ' ').asString

/-- 要素を右寄せして、Tensor 内の表示幅を揃える。 -/
private def padLeft (width : Nat) (value : String) : String :=
  spaces (width - value.length) ++ value

/-- shape に従って一次元の値を NumPy 風の多次元表現にする。 -/
private def formatValues (dims : List Nat) (values : List String) (depth : Nat) : String :=
  match dims with
  | [] => values.head?.getD ""
  | dim :: rest =>
      let chunkSize := rest.foldl (· * ·) 1
      let chunks := (List.range dim).map fun i =>
        formatValues rest ((values.drop (i * chunkSize)).take chunkSize) (depth + 1)
      let separator :=
        if rest.isEmpty then
          " "
        else
          (List.replicate rest.length '\n').asString ++ spaces (depth + 1)
      "[" ++ String.intercalate separator chunks ++ "]"
termination_by dims.length

/-- Tensor のデータを NumPy 風の多次元文字列に変換する。 -/
def toString [ToString α] {rank : Nat} {shape : Vector Nat rank}
    (t : Tensor α shape) : String :=
  let values := t.data.toList.map fun value => s!"{value}"
  let width := values.foldl (fun current value => max current value.length) 0
  let paddedValues := values.map (padLeft width)
  formatValues shape.toList paddedValues 0

/-- `IO.println tensor` のように Tensor を直接表示できるようにする。 -/
instance [ToString α] {rank : Nat} {shape : Vector Nat rank} :
    ToString (Tensor α shape) where
  toString := Tensor.toString

/-- indexがshapeの範囲内であることを保証する -/
def index_in_bounds {rank : Nat} (shape : Vector Nat rank) (index : Vector Nat rank) : Prop :=
  ∀ i : Fin rank, index.get i < shape.get i

/-- shape の各次元について範囲内であることが保証された Tensor 添字。 -/
structure Index {rank : Nat} (shape : Vector Nat rank) where
  values : Vector Nat rank
  isValid : index_in_bounds shape values

/-- 具体的な Tensor 添字の境界条件を自動証明する。 -/
macro "tensor_bounds" : tactic =>
  `(tactic|
    simp [DL.Tensor.index_in_bounds, Fin.forall_fin_succ, Vector.get] <;>
    omega)

/--
Vector から安全な `Tensor.Index` を作る。
境界条件を `tensor_bounds` で証明できなければコンパイルエラーになる。
-/
syntax "tensor_index% " term:arg : term
macro_rules
  | `(tensor_index% $values) =>
      `(DL.Tensor.Index.mk $values (by tensor_bounds))

/-- indexを1次元アクセスに変換する -/
def to_flat_index {rank : Nat} (shape : Vector Nat rank) (index : Vector Nat rank) : Nat :=
  (shape.zip index).foldl (fun flat di => flat * di.1 + di.2) 0

/-- to_flat_indexがデータのサイズ未満であることを保証する -/
theorem to_flat_index_lt_size
    {rank : Nat}
    (shape index : Vector Nat rank)
    (h : index_in_bounds shape index) :
    to_flat_index shape index < shape.foldl (· * ·) 1 := by

  -- zip の各要素 (dim, idx) について idx < dim
  have hzip :
      ∀ di : Nat × Nat, di ∈ shape.zip index → di.2 < di.1 := by
    intro di hdi
    -- zipされているので、shapeとindexの同じ位置の要素を取り出す
    obtain ⟨i, hi, hget⟩ := Vector.getElem_of_mem hdi
    have hb := h ⟨i, hi⟩
    have hb' : (shape.zip index)[i].2 < (shape.zip index)[i].1 := by
      simp [Vector.zip]
      exact hb
    rw [← hget]
    exact hb'

  -- flat index と、そこまでの shape の積を同時に fold する
  have hfold :
      (shape.zip index).foldl (fun flat di => flat * di.1 + di.2) 0
        <
      (shape.zip index).foldl (fun size di => size * di.1) 1 := by
    apply Vector.foldl_rel (r := fun x y : Nat => x < y)
    . omega
    · intro di hdi flat size hflat

      have hidx : di.2 < di.1 := hzip di hdi

      calc
        flat * di.1 + di.2
            < flat * di.1 + di.1 :=
              Nat.add_lt_add_left hidx (flat * di.1)
        _ = (flat + 1) * di.1 := by simp [Nat.add_mul]
        _ ≤ size * di.1 := Nat.mul_le_mul_right di.1 (Nat.succ_le_of_lt hflat)

  -- zip の first projection は元の shape
  have hfst :
      (shape.zip index).map (fun di => di.1) = shape := by
    ext i hi
    simp [Vector.zip]

  -- したがって右側の fold は shape 全体の積
  have hprod :
      (shape.zip index).foldl
          (fun size di => size * di.1) 1
        =
      shape.foldl (fun x y => x * y) 1 := by
    rw [← Vector.foldl_map]
    rw [hfst]

  unfold to_flat_index
  rw [hprod] at hfold
  exact hfold

/-- Tensor型からデータの取得 -/
def get {α : Type} {rank : Nat} {shape : Vector Nat rank}
    (t : Tensor α shape)
    (index : Index shape) : α :=
  let flat_index := to_flat_index shape index.values
  let hsize : flat_index < shape.foldl (· * ·) 1 :=
    to_flat_index_lt_size shape index.values index.isValid
  have : flat_index < t.data.size := by
    rw [t.hsize]
    exact hsize

  t.data[flat_index]

/-- 安全な `Tensor.Index` を使った `tensor[index]` 記法。 -/
instance {α : Type} {rank : Nat} {shape : Vector Nat rank} :
    GetElem (Tensor α shape) (Index shape) α (fun _ _ => True) where
  getElem t index _ := get t index

/-- Tensor型へのデータの書き込み -/
def set {α : Type} {rank : Nat} {shape : Vector Nat rank}
    (t : Tensor α shape)
    (index : Index shape)
    (value : α) : Tensor α shape :=
  let flat_index := to_flat_index shape index.values
  let hsize : flat_index < shape.foldl (· * ·) 1 :=
    to_flat_index_lt_size shape index.values index.isValid
  have : flat_index < t.data.size := by
    rw [t.hsize]
    exact hsize

  let new_data := t.data.set flat_index value
  have hsize' : new_data.size = shape.foldl (· * ·) 1 := by
    rw [Array.size_set]
    exact t.hsize
  { data := new_data, hsize := hsize' }

/-- shapeのサイズが0であること-/
def shape_is_zero {rank : Nat} (shape : Vector Nat rank) : Prop :=
  shape.foldl (· * ·) 1 = 0

/-- shapeの合計サイズが0でないことと各要素が0でないことは同値-/
theorem shape_nonzero_iff {rank : Nat} (shape : Vector Nat rank) :
    ¬ shape_is_zero shape ↔ ∀ i : Fin rank, shape.get i ≠ 0 := by
  unfold shape_is_zero
  constructor
  .
    intro hsize i
    by_contra h
    have hprod : shape.foldl (· * ·) 1 = 0 := by
      apply DL.Vector.foldl_mul_eq_zero_of_mem shape 1
      rw [← h]
      apply Vector.getElem_mem
    contradiction
  .
    exact DL.Vector.foldl_mul_ne_zero rank shape

/-- 1次元indexから多次元indexへ変換する。 -/
def to_multi_index {rank : Nat} (shape : Vector Nat rank) (flat_index : Nat) (isLt : flat_index < shape.foldl (· * ·) 1) : Index shape :=
  -- 最初にshapeの各要素が0でないことを確認する
  have hsize : ∀ i : Fin rank, shape.get i ≠ 0 := by
    apply (shape_nonzero_iff shape).mp
    simp [shape_is_zero]
    omega
  have hsize_nonzero : ∀ i : Fin rank, shape.get i > 0 := by
    intro i
    have h := hsize i
    omega
  let values := Vector.ofFn fun i =>
    let stride := (shape.toList.drop (i.val + 1)).foldl (· * ·) 1
    (flat_index / stride) % shape.get i
  {
    values := values
    isValid := by
      intro i
      change values[i.val] < shape.get i
      simp [values]
      exact Nat.mod_lt _ (hsize_nonzero i)
  }

end Tensor
end DL
