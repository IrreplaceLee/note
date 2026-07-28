# 张量的底层存储：storage、stride、view 与 contiguous

> 整理日期：2026-07-29 00:20
> 关联书籍：《PyTorch 深度学习实战》（第 3 章）

## 本轮学到的核心知识

- **底层存储（storage）：张量在内存里其实是「拉直」的一维连续数组**。不管张量看起来是几维，数据都是一条线排下来的。`points.storage()` 能看到这条线。
  - 一个 3×2 的 `[[4,1],[5,3],[2,1]]`，底层就是 `4 1 5 3 2 1` 六个数按顺序排。

- **张量 = 一份存储 + 三个「怎么看」的参数**：
  - **storage_offset（偏移）**：从存储的第几个位置开始看。`points[1]`（第 2 行）的偏移是 2。
  - **size / shape（形状）**：每一维多长。
  - **stride（步长）**：**沿某一维走 1 步，在底层一维存储里要跨过几个元素**。3×2 张量的 stride 是 `(2, 1)`：往下走一行跨 2 个数，往右走一列跨 1 个数。

- **视图（view）vs 副本（clone）——最容易踩的坑**：
  - `second = points[1]` 取出来是**视图**，和原张量**共享同一份存储**。改视图会连原张量一起改（`second[0] = 10` 后 `points` 也变了）。
  - 想要独立数据就用 `.clone()`：`points[1].clone()` 是复制出的副本，改它不影响原张量。

- **转置（transpose / `.t()`）只是「换个角度看」，不搬数据**：
  - `points.t()` 把 3×2 看成 2×3，但底层存储一个字节都没动，只是把 stride 从 `(2,1)` 换成 `(1,2)`。
  - 用 `id(a.storage()) == id(b.storage())` 可验证二者共享存储 → `True`。
  - `.transpose(dim0, dim1)` 是通用版，能对任意两维互换（三维、四维都行）；`.t()` 只用于二维。

- **contiguous（连续）**：
  - `is_contiguous()` 判断「数据的逻辑顺序」和「内存实际顺序」是否一致。原始张量为 `True`，转置后为 `False`。
  - `.contiguous()` 会**真正重排内存**，生成一份数据连续的新张量（stride 也变回正常）。
  - 为什么要它：有些操作（如 `.view()`）要求张量连续，转置后直接用会报错，先 `.contiguous()` 再用。

## 我踩过的坑 / 薄弱项

- **误区**：以为 `points[1]` 是复制出一行、改它不影响原数据。**正解**：它是视图、共享存储，改视图会改原张量；要独立就 `.clone()`。
- **误区**：以为转置会重新生成一份行列互换的数据。**正解**：转置只改 stride、不动存储，是「零拷贝」的；只有 `.contiguous()` 才真正搬数据。
- **薄弱项**：不理解「几维张量在内存里到底怎么存」。答案是——**永远是一维连续数组，靠 stride 换算出多维的样子**。

## 可运行的代码示例

```python
# ===== Kaggle Notebook 可直接运行 =====
import torch
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])  # 3×2

# 1) 底层存储是拉直的一维数组
print(points.storage())        # 4 1 5 3 2 1

# 2) 偏移 / 形状 / 步长
second = points[1]             # 取第2行，是视图
print(second.storage_offset()) # 2   从存储第3个位置开始看
print(second.shape)            # torch.Size([2])
print(points.stride())         # (2, 1)

# 3) 视图 vs 副本
second[0] = 10.0
print(points)                  # 原张量第2行也变成 10. —— 共享存储

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
sp = points[1].clone()         # 副本，独立
sp[0] = 99.0
print(points)                  # 原张量不受影响

# 4) 转置只改 stride，不搬数据
pt = points.t()
print(pt.stride())             # (1, 2)  正好反过来
print(id(points.storage()) == id(pt.storage()))  # True 共享同一份存储

# 5) contiguous 才真正重排内存
print(pt.is_contiguous())              # False
print(pt.contiguous().stride())        # (3, 1)  连续化后步长正常
```

## 延伸提示

- **前置**：先理解 shape 与切片降维见 [张量切片与范围索引](张量切片与范围索引-0020.md)；张量是什么见 [张量 Tensor 入门](../20260724/张量Tensor入门-2222.md)。
- **实践提醒**：以后自己写代码时，如果「改了 A，B 莫名其妙也变了」，八成是 A、B 共享存储（切片/视图），需要 `.clone()`；遇到 `view() 报错 not contiguous`，先加 `.contiguous()`。
- **和研究方向的联系**：MRI 体数据张量很大，理解「视图零拷贝」能帮你写出省内存的预处理代码——对切片、翻转、转置这些操作，PyTorch 尽量不复制数据。将来做多模态融合时，不同模态张量对齐、维度重排（permute/transpose）也依赖 stride 这套机制。
