# torch 库是什么 · 核心子模块与训练流程

> 整理日期：2026-07-24 22:14
> 关联书籍：《PyTorch 深度学习实战》

## 本轮学到的核心知识

- **`torch` 就是 PyTorch 库在代码里的名字**。写 `import torch` 后，所有功能都通过 `torch.xxx` 调用。可以把它理解成一个「深度学习工具箱」。
  - **PyTorch** 是框架的正式名字（官网、书上用）；**torch** 是它在代码里的包名（import 时用）。两者指同一个东西。
  - 名字来历：前身是 Lua 语言写的老框架 Torch，PyTorch 是它的 Python 版本，保留了 torch 这个名字。

- **torch 下最常打交道的 5 个部分**（按学习顺序）：

  | 模块 | 干什么的 | 生活类比 |
  |------|---------|---------|
  | **Tensor（张量）** | 装数字的多维容器，算得快、能上 GPU | 食材（所有数据的基本形态） |
  | **torch.nn** | 搭神经网络的零件（各种层） | 乐高积木块 |
  | **torch.optim** | 训练时调整模型参数 | 帮你纠错的教练 |
  | **torch.utils.data** | 批量喂数据（Dataset / DataLoader） | 传送带 |
  | **torch.cuda** | 调用 GPU 加速 | 涡轮增压开关 |

- **一次完整训练里，它们怎么串起来**：
  1. `torch.utils.data` 把数据打包成一批
  2. `torch.nn` 搭好网络
  3. 数据先变成 Tensor
  4.（可选）`torch.cuda` 把模型和数据搬到显卡
  5. 前向计算 → 得到预测
  6. 算误差（loss）：预测和真实答案差多少
  7. `torch.optim` 根据误差更新参数
  8. 回到第 5 步反复循环 —— 这就是「训练」

## 我踩过的坑 / 薄弱项

- **误区**：以为 `torch` 和 `PyTorch` 是两个不同的东西。
  **正解**：同一个东西，一个是品牌名，一个是代码里的包名。
- **薄弱项**：对「库 / 模块 / 子模块」这种 Python 基础概念还不熟。`torch` 是一个大包，`torch.nn`、`torch.optim` 是它下面的子模块，用点号 `.` 一层层访问。
- 目前在书的**第一部分**，重点先吃透 **Tensor**（地基），其他 4 个模块建立在它之上，第二部分（肺结节检测项目）会全部一起上场。

## 可运行的代码示例

```python
# ===== 在 Kaggle Notebook 里直接运行 =====
import torch                      # 导入 PyTorch，之后才能用 torch.xxx

# 1) 查看版本，确认 torch 已就绪
print("PyTorch 版本：", torch.__version__)

# 2) 用 torch 造一个张量（最核心的东西）
x = torch.tensor([1, 2, 3])
print("张量 x：", x)
print("形状：", x.shape)          # torch.Size([3])

# 3) 看看几个常用子模块是否都能访问到
import torch.nn as nn             # 搭网络的零件
import torch.optim as optim       # 优化器
from torch.utils.data import DataLoader  # 喂数据

print("nn 模块：", nn)            # 能打印出来就说明子模块正常
print("GPU 是否可用：", torch.cuda.is_available())  # 需在 Settings→Accelerator 选 GPU 才为 True
```

## 延伸提示

- **下一步**：先把 Tensor 学扎实——建议接着掌握「从张量里取数字」（索引和切片），这是最高频操作。对应书第 3 章。
- **和研究方向的联系**：
  - **Tensor → MRI 影像的数据表示**。一张 MRI 本质就是多维张量（3D 体数据甚至更高维），将来做 NPSLE 影像处理时，读进来的每一张图都要先变成 Tensor 才能送进模型。这正是「了解 MRI 影像特性及处理方式」这条主线的技术入口。
  - **torch.nn / torch.optim → 建模与训练的通用底座**。将来无论用 Transformer 还是 GCN 建模 NPSLE，训练流程都是这同一套（数据→网络→算误差→更新参数）。现在把流程理顺，以后换什么模型都通用。
  - **torch.cuda → 医学影像算得动的前提**。MRI 数据量大，没有 GPU 加速训练几乎跑不动，所以 `torch.cuda` 这套将来是刚需。
