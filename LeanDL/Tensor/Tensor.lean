import Init.Data.Vector.Lemmas

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

/-- Tensor の shape と一次元化されたデータを文字列に変換する。 -/
def toString [ToString α] {rank : Nat} {shape : Vector Nat rank}
    (t : Tensor α shape) : String :=
  s!"Tensor(shape := {shape.toArray}, data := {t.data})"

/-- `IO.println tensor` のように Tensor を直接表示できるようにする。 -/
instance [ToString α] {rank : Nat} {shape : Vector Nat rank} :
    ToString (Tensor α shape) where
  toString := Tensor.toString

/-- indexがshapeの範囲内であることを保証する -/
def index_in_bounds {rank : Nat} (shape : Vector Nat rank) (index : Vector Nat rank) : Prop :=
  ∀ i : Fin rank, index.get i < shape.get i

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
    (index : Vector Nat rank)
    (h : index_in_bounds shape index) : α :=
  let flat_index := to_flat_index shape index
  let hsize : flat_index < shape.foldl (· * ·) 1 := to_flat_index_lt_size shape index h
  have : flat_index < t.data.size := by
    rw [t.hsize]
    exact hsize

  t.data[flat_index]

/-- Tensor型へのデータの書き込み -/
def set {α : Type} {rank : Nat} {shape : Vector Nat rank}
    (t : Tensor α shape)
    (index : Vector Nat rank)
    (h : index_in_bounds shape index)
    (value : α) : Tensor α shape :=
  let flat_index := to_flat_index shape index
  let hsize : flat_index < shape.foldl (· * ·) 1 := to_flat_index_lt_size shape index h
  have : flat_index < t.data.size := by
    rw [t.hsize]
    exact hsize

  let new_data := t.data.set flat_index value
  have hsize' : new_data.size = shape.foldl (· * ·) 1 := by
    rw [Array.size_set]
    exact t.hsize
  { data := new_data, hsize := hsize' }


end Tensor
end DL
