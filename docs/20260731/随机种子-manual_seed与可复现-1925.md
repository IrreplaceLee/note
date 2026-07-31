# 随机种子：manual_seed 与 initial_seed 实现可复现

> 整理日期：2026-07-31 19:25
> 关联书籍：《PyTorch 深度学习实战》

## 本轮学到的核心知识

- **为什么需要「随机种子（seed）」**
  - PyTorch 里很多操作是随机的：初始化网络权重、随机采样、打乱数据、数据增强等。
  - 不固定种子 → 每次运行结果都不一样 → 实验无法复现、调 bug 时现象飘忽。
  - 固定种子 → 每次运行得到**完全相同**的随机结果 → 便于调试、便于论文复现。

- **`torch.random.manual_seed(seed)`：手动设置种子**
  - 传入一个整数作为种子，之后产生的随机数序列就被「锁定」了。
  - 只要种子相同，`torch.rand()` 等函数每次跑出来的值都一样。
  - 一般在**代码最开头设置一次**即可。
  - 等价写法：`torch.manual_seed(seed)`（更常见的简写，效果相同）。

- **`torch.random.initial_seed()`：查看当前用的种子**
  - 返回当前生效的种子值（一个整数）。
  - 主要用于**调试 / 记录**：想知道「这次实验用的是哪个种子」时查看它。
  - 如果从没手动设过种子，它会返回一个系统随机生成的默认种子。

- **种子选什么数字**
  - 任何整数都行（`0`、`42`、`3407` 都可以）。
  - 重点不是选哪个数，而是**把用过的种子记录下来**，下次复现时用同一个。
  - 坊间有人测过 `3407` 在不少任务上比较稳，可当彩蛋试试，但不必迷信。

## 我踩过的坑 / 薄弱项

- **只固定 PyTorch 的种子还不够**：一个训练流程往往同时用到 PyTorch、NumPy、Python 自带 `random` 三套随机源。只调 `torch.manual_seed` 的话，NumPy / random 产生的随机（如数据打乱、增强）仍然每次都变，结果依旧不可复现。
  → 正确做法是**三套一起固定**（见下方代码的 `set_seed`）。

- **`manual_seed` 和 `initial_seed` 别搞混**：一个是「设置」（写入），一个是「查看」（读取）。设种子用 `manual_seed`，查种子用 `initial_seed`。

## 可运行的代码示例

```python
import torch

# 基本用法：设种子后，两次 rand 结果完全相同
torch.manual_seed(42)
print(torch.rand(3))      # 例如 tensor([0.8823, 0.9150, 0.3829])

torch.manual_seed(42)     # 重置回同一个种子
print(torch.rand(3))      # 和上面完全一样

# 查看当前种子
torch.manual_seed(999)
print(torch.random.initial_seed())   # 999
```

```python
# 实战推荐：一次性固定所有随机源，保证实验可复现
import torch
import numpy as np
import random

def set_seed(seed=42):
    """固定所有随机数种子，保证结果可复现。"""
    torch.manual_seed(seed)                     # PyTorch CPU
    torch.cuda.manual_seed(seed)                # PyTorch 当前 GPU
    torch.cuda.manual_seed_all(seed)            # 多 GPU 情况
    np.random.seed(seed)                        # NumPy
    random.seed(seed)                           # Python 自带 random
    torch.backends.cudnn.deterministic = True   # cuDNN 走确定性算法
    torch.backends.cudnn.benchmark = False      # 关掉自动寻优（否则可能引入随机）

# 在整个训练脚本的最开头调用一次
set_seed(42)
# ……接下来写你的数据加载、建模、训练代码……
```

## 延伸提示

- 在 Kaggle 上跑实验时，建议每个 Notebook 开头都调一次 `set_seed(...)`，并把种子值写在标题或注释里，方便日后复现同一结果。
- `torch.backends.cudnn.deterministic = True` 会让 GPU 走确定性算法，可复现性变好，但速度可能略降；追求极致速度、不在乎复现时可以关掉。
- **与研究任务的联系**：SLE / NPSLE 方向数据本就稀缺（小样本），实验结论更依赖「可复现」来站得住脚——同一份数据、同一个种子，别人（和几个月后的你自己）才能跑出一致结果。写论文/做对比实验时，固定并汇报随机种子是基本规范。做少样本、交叉验证划分时尤其要注意用种子锁定每一折的划分，否则每次跑出来的指标都在飘。
