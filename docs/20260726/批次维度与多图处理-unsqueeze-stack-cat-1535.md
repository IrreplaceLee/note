# 批次维度与多图处理（unsqueeze / stack / cat）

> 整理日期：2026-07-26 15:35
> 关联书籍：《PyTorch 深度学习实战》第 2 章

## 🌟 只记这四条就够

1. **神经网络永远吃 4 维输入：`(N, C, H, W)`** —— N=批次几张图、C=通道数(RGB是3)、H=高、W=宽。
2. **一张图经过 `preprocess` 后是 3 维 `(3, 224, 224)`——差一个 batch 维。**
3. **一张图 → 一批：用 `unsqueeze(0)` 在最前面套一层。**
4. **多张图 → 一批：每张单独 `preprocess`，再用 `torch.stack` 堆起来。**

## 形状变化图（建议背下来）

```
单张图：
原始图片 --preprocess--> (3, 224, 224) --unsqueeze(0)--> (1, 3, 224, 224) ✅

多张图：
多张原图 --每张分别 preprocess--> [(3,224,224), (3,224,224), ...]
        --torch.stack([...], dim=0)--> (N, 3, 224, 224) ✅
```

## 为什么必须有 batch 维

- PyTorch 模型是按「批量」设计的，**不接受单张图**。训练时通常一次喂一批（如 32 张）进 GPU 并行算，效率高，所以第一维默认就是 batch 维。
- 即使只推理一张图，也要**假装是「批次里只有 1 张」**，凑够 4 维。
- **打饭比喻**：网络是食堂窗口，规定必须用**托盘**递餐——哪怕托盘上只有 1 份饭，托盘也不能省。`unsqueeze(0)` 就是套托盘。

## 三个操作的区别（最容易混）

| 操作 | 用在什么时候 | 一句话记忆 |
|---|---|---|
| **`x.unsqueeze(0)`** | 只有 1 张图（3 维） | **加一维**——凭空插入 batch 维 |
| **`torch.stack([a,b], dim=0)`** | 多张图，还没有 batch 维（都是 3 维） | **堆起来**——造新维并叠加 |
| **`torch.cat([x,y], dim=0)`** | 多个已带 batch 维的张量（都是 4 维） | **拼长**——沿已有维度接起来 |

判断口诀：**单张→unsqueeze；多张(3维)→stack；多批(4维)→cat**。

## 我踩过的坑 / 薄弱项

- **`torch.stack` 要求所有张量形状完全一致**，否则报错。幸好 `preprocess` 里有 Resize+CenterCrop，出来都是 `(3, 224, 224)`，能直接堆——**这也是 preprocess 必须存在的另一个理由**。
- 若跳过 preprocess 直接堆原图（尺寸不一），`stack` 会报形状不匹配的错。
- 分不清 `stack` 和 `cat`：`stack` 会**新建**一个维度，`cat` 是沿**已有**维度拼接。

## 可运行的代码示例

```python
import torch
from PIL import Image

# ---------- 单张图推理 ----------
img = Image.open("dog.jpg")
img_t = preprocess(img)                 # (3, 224, 224)
batch_t = torch.unsqueeze(img_t, 0)     # (1, 3, 224, 224)  等价 img_t.unsqueeze(0)
out = resnet(batch_t)                   # (1, 1000)

# ---------- 多张图推理 ----------
paths = ["cat.jpg", "dog.jpg", "bird.jpg"]
imgs = [Image.open(p) for p in paths]
batch_t = torch.stack([preprocess(im) for im in imgs], dim=0)  # (3, 3, 224, 224)
out = resnet(batch_t)                   # (3, 1000)  每行一张图的结果

# ---------- stack vs cat 对比 ----------
a = torch.randn(3, 224, 224); b = torch.randn(3, 224, 224)
print(torch.stack([a, b], dim=0).shape)          # (2, 3, 224, 224)  新建维度

x = torch.randn(1, 3, 224, 224); y = torch.randn(1, 3, 224, 224)
print(torch.cat([x, y], dim=0).shape)            # (2, 3, 224, 224)  拼长已有维
```

## 延伸提示

- 这是 PyTorch 数据喂入的**通用规范**，做 MRI 时会常用：3D 影像的 batch 形状是 `(N, C, D, H, W)`（多一个深度维 D）。
- 后面学 `DataLoader` 时，它会自动帮你把多个样本 `stack` 成一个 batch，就不用手动写了；但理解手动的过程有助于 debug。
- 相关笔记：[图像预处理流水线与张量表示](图像预处理流水线与张量表示-1535.md)、[张量索引与取值](../20260725/张量索引与取值-0040.md)。