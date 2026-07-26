# 解读网络输出（torch.max 与 softmax）

> 整理日期：2026-07-26 15:35
> 关联书籍：《PyTorch 深度学习实战》第 2 章

## 本轮学到的核心知识

推理后 `out = resnet(batch_t)` 的形状是 `(1, 1000)`：第 0 维是 batch（第几张图），第 1 维是 **1000 个类别的原始分数**（ImageNet 有 1000 类）。这些分数可能是负数、可能大于 1，本身没意义，要做两件事翻译成人话——**找出最高分**、**转成概率**。

- **读类别名字**：`imagenet_classes.txt` 每行一个类别名，读进列表 `labels`，网络输出的是下标（如 207），靠这个列表把下标翻译成名字（"golden retriever"）。
- **`torch.max(out, 1)` 找最高分**：沿第 1 维（1000 类那维）找最大值，返回**两个东西**——最大值本身、最大值的位置（下标）。`_, index = ...` 里的 `_` 是 Python 惯例，表示「这个变量我不要」（这里不要分数只要下标）。
- **`softmax` 把分数变概率**：把任意数字压缩到 `0~1` 之间，并保证 1000 个数**加起来等于 1**。乘 100 就变成百分比。
- **`.item()`**：把「单个数的张量」`tensor(96.29)` 转成普通 Python 数字 `96.29`，打印更干净。

## 逐行拆解

```python
_, index = torch.max(out, 1)
# out 沿第1维找最大值 → index = tensor([207])（最可能的类别下标）

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# softmax(out, dim=1): (1,1000) 分数 → (1,1000) 概率，加起来=1
# [0]: 取第0张图，形状变 (1000,)
# *100: 0~1 变成 0~100 百分比

labels[index[0]], percentage[index[0]].item()
# index[0] 取出下标值 207
# labels[207] → "golden retriever"（查名字）
# percentage[207].item() → 96.29（查概率并转普通数字）
# 结果元组：("golden retriever", 96.29)
```

## 我踩过的坑 / 薄弱项

- 不理解 `torch.max` 为什么返回两个值：它同时给「最大值」和「最大值下标」，取下标常用 `_, index = torch.max(x, dim)` 忽略掉值。
- 分不清 `out` 的原始分数和概率：**原始分数（logits）不能直接当概率**，必须过 `softmax` 才能加起来等于 1、才好说「多少 % 确信」。
- `dim` 用错：batch 维是 0，类别维是 1，找「哪个类别分最高」要沿 `dim=1`。
- 忘了 `.item()`：不转的话打印出来是 `tensor(96.29)` 而不是干净的 `96.29`；`.item()` 只对**单个元素**的张量有效。

## 可运行的代码示例

```python
import torch

# 假设已有 out = resnet(batch_t)，形状 (1, 1000)

# 1. 读类别名
with open('../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]   # 长度 1000 的列表

# 2. 找最高分类别
_, index = torch.max(out, 1)                             # index = tensor([207])

# 3. 分数转百分比概率
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# 4. 输出「名字 + 概率」
print(labels[index[0]], percentage[index[0]].item())    # golden retriever 96.29

# 进阶：看 Top-5 最可能的类别
_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
```

## 延伸提示

- `softmax` + `argmax`（即 `torch.max` 取下标）是所有**分类任务**的标准收尾套路，你未来做 SLE/NPSLE 的「有病 / 无病」或分型分类时同样用它。
- 概念小结表：`torch.max(x, dim)` 返回 (值, 下标)；`_` 是占位符；`softmax` 输出加起来=1；`.item()` 张量转数字；`dim=1` 沿类别维操作。
- 相关笔记：[[批次维度与多图处理-unsqueeze-stack-cat-1535]]、[[张量索引与取值-0040]]。
```