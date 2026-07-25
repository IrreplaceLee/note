# PyTorch 环境与 import 基础（Kaggle）

> 整理日期：2026-07-24 22:22
> 关联书籍：《PyTorch 深度学习实战》

## 本轮学到的核心知识

- **用任何库之前必须先 `import`**。直接写 `torch.cuda.is_available()` 会报 `NameError: name 'torch' is not defined`，意思是当前 Python 环境里根本没有 `torch` 这个名字，和有没有 GPU 无关。要先 `import torch`。
- **每个 Notebook 会话都要重新导入**。如果重启了 kernel（Run → Restart），之前 import 的东西全部清空，需要重新运行导入单元格。
- **`torch.cuda.is_available()` 要返回 `True` 的前提**：在 Kaggle 右侧 Settings → Accelerator 里选中 GPU（如 GPU T4 x2），否则即使 import 成功也返回 `False`。
- **「Notebook 用的 torch 和终端用的 torch 是不是同一个」是什么意思**：一台机器上常有多个 Python 环境（系统自带、conda base、各种虚拟环境）。Jupyter 的 kernel 绑定某一个环境，终端里的 `python` 可能是另一个，于是会出现「一边能 import、一边报错」或版本不一致。
  - 判断方法：分别在 Notebook 和终端打印 `sys.executable`（解释器路径）、`torch.__file__`（库文件位置）、`torch.__version__`（版本）比对是否一致。
  - **对 Kaggle 而言不用担心**：它是托管环境，没有独立本地终端，kernel 用的就是预装好的那个 conda 环境（路径类似 `/opt/conda/...`）。

## 我踩过的坑 / 薄弱项

- **误区**：以为 `torch.cuda.is_available()` 报错是「没有 GPU / CUDA 问题」。
  **正解**：`NameError` 是「没导入 torch」，纯 Python 基础问题，先 `import torch`。
- **薄弱项**：对「import 是干什么的、kernel 重启会清空变量、不同 Python 环境」这些基础机制还不熟。

## 可运行的代码示例

```python
# ===== 在 Kaggle Notebook 里直接运行 =====
import torch

# 1) 确认 torch 已就绪 + 版本
print("PyTorch 版本：", torch.__version__)

# 2) 检查 GPU 是否可用（需先在 Settings→Accelerator 选 GPU）
print("GPU 是否可用：", torch.cuda.is_available())
print("设备名：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# 3) 确认当前环境（Kaggle 上会指向 /opt/conda/... ）
import sys
print("解释器路径：", sys.executable)
print("torch 文件位置：", torch.__file__)
```

## 延伸提示

- Kaggle 默认已预装 PyTorch，通常不需要自己 `pip install`。
- **和研究方向的联系**：将来做 NPSLE 的 MRI 训练，数据量大、必须用 GPU，所以「确认 GPU 可用」会是每次开工的固定第一步。养成先 `import torch` + 检查 `cuda.is_available()` 的习惯。
