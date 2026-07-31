# 张量数据类型判断：dtype、小数点与 type() 的区别

> 整理日期：2026-07-31 19:25
> 关联书籍：《PyTorch 深度学习实战》

## 本轮学到的核心知识

- **看有没有小数点，就能判断张量是整数还是浮点数**
  - 打印结果里数字带小数点（如 `1.`，其实就是 `1.0`）→ 浮点型（默认 `float32`）
  - 数字不带小数点（如 `1`）→ 整数型（默认 `int64`）
  - 小数点是**数字本身的写法**，只要是浮点型就一定会显示，永远不会省略。

- **不同创建方式，默认类型不一样**
  - `torch.ones(2, 3)`：`ones` / `zeros` / `rand` 造出来默认是 **float32** → 打印是 `1.`
  - `torch.tensor([[1, 2], [3, 4]])`：你填的是整数（没写小数点），PyTorch 自动推断成 **int64** → 打印是 `1`
  - 若写成 `torch.tensor([[1., 2.]])`（带小数点），才会变成 float32

- **`ones_like` / `zeros_like` 会继承原张量的「形状 + 数据类型」**
  - `torch.ones_like(t2)`：照着 t2 的形状和 dtype 造全 1 张量
  - t2 是 int64，所以 t3 也是 int64（不是 float32）

- **`type()` 和 `.dtype` 是两回事，判断数据类型要用 `.dtype`**
  - `type(t)` 是 Python 内置函数，只告诉你「这是个 `torch.Tensor` 对象」，**永远返回 `<class 'torch.Tensor'>`**，看不出里面装的是 int 还是 float。
  - `t.dtype` 才是张量真正的数据类型（`torch.float32` / `torch.int64` 等）。

## 我踩过的坑 / 薄弱项

- **误区一**：以为 `type(t2)` 能打印出 int 还是 float。
  → 实际上 `type()` 对任何张量都只返回 `torch.Tensor`。想看数据类型必须用 `.dtype`。

- **误区二**：以为「float32 是默认类型，所以不打印，因此没小数点」，把 t2、t3 也当成了 float32。
  → 这里混淆了**两个不同的东西**：
    1. **`dtype=...` 这个类型标签**：`print(t)` 时，float32 是默认类型会被省略不显示；非默认类型有时会标 `dtype=torch.int64`。这个标签**不能**用来可靠判断类型。
    2. **数值里的小数点**：这是数字本身的形态，浮点型一定显示 `1.`，整数型一定显示 `1`，**永远不省略**。
  → 所以 t2、t3 打印是 `1`（没小数点），说明它们**根本就是 int64**，而不是「float32 被省略了小数点」。float32 就算省掉 dtype 标签，小数点也照样在。

- **一句话记牢**：看小数点最准 —— 有小数点=浮点，没小数点=整数；`dtype=` 标签不可靠，`type()` 更是完全没用。

## 可运行的代码示例

```python
import torch

# t1: ones 默认 float32 → 打印带小数点
t1 = torch.ones(2, 3)
print(f't1:\n{t1}')
print('t1.dtype =', t1.dtype)       # torch.float32
print('type(t1) =', type(t1))       # <class 'torch.Tensor'>（看不出类型）
print('-' * 40)

# t2: 填的是整数 → 推断成 int64 → 打印无小数点
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2:\n{t2}')
print('t2.dtype =', t2.dtype)       # torch.int64
print('-' * 40)

# t3: ones_like 继承 t2 的形状和 dtype → 也是 int64
t3 = torch.ones_like(t2)
print(f't3:\n{t3}')
print('t3.dtype =', t3.dtype)       # torch.int64
print('-' * 40)

# 对比：填带小数点的数 → float32
t4 = torch.tensor([[1., 2.], [3., 4.]])
print(f't4:\n{t4}')
print('t4.dtype =', t4.dtype)       # torch.float32

# 需要指定类型时，用 dtype 参数
t5 = torch.ones(2, 3, dtype=torch.int64)   # 强制整数
print('t5.dtype =', t5.dtype)              # torch.int64
```

## 延伸提示

- 深度学习里绝大多数运算（尤其是要求梯度、送进神经网络的输入）都需要 **float32**。如果不小心造了 int64 张量再送进网络，常会报类型不匹配的错。转换方法：`t = t.float()` 或 `t = t.to(torch.float32)`。
- 整数张量还有一个常见坑：对 `int` / `uint8` 类型直接调用 `.mean()` 会报错（均值可能是小数，整数存不下）。这类报错已单独记录在 [问题排查与配置记录](../问题排查与配置记录.md)。
- 张量的数据类型、设备（CPU/GPU）、原地操作等更系统的内容，见 [张量类型-设备GPU-原地操作与存取](../20260729/张量类型-设备GPU-原地操作与存取-0020.md)。
- **与研究任务的联系**：以后处理 MRI 影像数据时，像素/体素值读进来常是整数（如 `uint8`、`int16`），但送进网络前几乎都要先归一化并转成 `float32`。养成「造完/读完张量先看一眼 `.dtype`」的习惯，能避免后面一大堆隐蔽的类型错误。
