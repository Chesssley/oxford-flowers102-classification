# Oxford Flowers102 Classification

使用 PyTorch 在 [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) **全部 102 类**上完成 CNN 图像分类课程实验，沿用官方 train / val / test 划分。项目包含自行实现的 SimpleCNN 基线、使用 ImageNet 预训练权重进行迁移学习的 ResNet18、数据检查、训练、独立评估、指标 JSON、图表及 [实验结果报告](reports/experiment-results.md)。当前正式实验中，ResNet18 的 test top-1 accuracy 为 **86.94%**。

## 数据与防泄漏约定

数据由 [`torchvision.datasets.Flowers102`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html) 下载到项目 `data/`。严格使用 Oxford 官方 `setid.mat` 中的 `train`（1020 张）、`val`（1020 张）、`test`（6149 张），共 8189 张、102 类。`train` 更新模型参数；`val` 选择最佳 checkpoint 和判断是否提前停止；`test` 只在正式训练结束后评估。本次没有根据 test 结果调参。

训练输入使用 `RandomResizedCrop(224, scale=(0.7, 1.0))`、水平翻转、10° 内随机旋转；验证和测试使用 `Resize(256)` 与 `CenterCrop(224)`。三者均使用 [ResNet18 官方 ImageNet 权重](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)的 RGB 均值和标准差进行归一化。增强只用于训练集。

## 项目结构

```text
.
├── AGENTS.md
├── .gitignore
├── README.md
├── requirements.txt
├── src/flowers102/       # 数据、模型、训练、评估、绘图、工具函数
├── scripts/              # inspect_data.py, train.py, evaluate.py, build_report.py
├── reports/experiment-results.md
├── .venv/                # 本地环境，Git 忽略
├── data/                 # 官方数据与预训练权重，Git 忽略
└── outputs/              # checkpoint、JSON、图表，Git 忽略
```

`data/` 和 `outputs/` 会由命令按需创建。所有命令均从仓库根目录运行；Windows 示例直接调用 `.venv` 内 Python，不需要激活环境，也不会向全局 Python 安装项目依赖。

## 环境重建

本次实测环境：Windows、64 位 Python 3.12.3、NVIDIA GeForce RTX 4070 SUPER、驱动 617.14。`requirements.txt` 固定 torch 2.13.0+cu130、torchvision 0.28.0+cu130、SciPy 1.18.1、NumPy 2.5.2、Matplotlib 3.11.2 等**直接依赖**。PyTorch wheel 来自 [官方 CUDA 13.0 索引](https://pytorch.org/get-started/previous-versions/)；[NVIDIA 文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)列出的 CUDA 13.x 最低驱动为 580。无需改动系统 CUDA Toolkit。代码在无 CUDA 时可运行于 CPU，但训练会更慢；本依赖清单指定的是 CUDA wheel。

若从新 checkout 重建，先创建 `.venv`。本项目完整路径较长，PyTorch wheel 内深层许可证文件曾触发 Windows `WinError 206`；安装时可用 Windows 官方支持的临时 `subst` 盘符缩短路径。`O:` 必须空闲，映射结束即撤销，实际文件仍在当前项目中：

```powershell
py -3.12 -m venv .venv
if (Test-Path 'O:\') { throw 'O: 已被占用，请改用其他空闲盘符' }
subst O: (Get-Location).Path
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
try {
    New-Item -ItemType Directory -Force O:\.venv\temp | Out-Null
    $env:TEMP = 'O:\.venv\temp'
    $env:TMP = $env:TEMP
    & O:\.venv\Scripts\python.exe -m pip install --no-cache-dir -r O:\requirements.txt
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
    subst O: /d
}
.\.venv\Scripts\python.exe -m pip check
```

本次实际沿用已有 `.venv`，没有删除或重建它。新增依赖通过该环境的 `python -m pip` 安装，`pip check` 和导入验证均已通过。预训练权重下载到项目 `data/torch-hub/`，Matplotlib 缓存及临时文件配置在项目 `.venv/` 内。

## 运行实验

首次下载并验证官方数据。命令检查官方 ID 无交叉、三种 split 的数量与标签范围、图像文件，以及随机样本的 `float32`、`3×224×224` 张量：

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py --download
```

正式训练前，两个模型各做少量 batch 的 smoke test，并在 **val** 上独立加载 smoke checkpoint。Smoke 产物保存在 `outputs/smoke/`，不作为最终结果：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --model simple-cnn --smoke-test --batch-size 32 --workers 0
.\.venv\Scripts\python.exe scripts\evaluate.py --model simple-cnn --split val --smoke-test --batch-size 32 --workers 0
.\.venv\Scripts\python.exe scripts\train.py --model resnet18 --smoke-test --batch-size 32 --workers 0
.\.venv\Scripts\python.exe scripts\evaluate.py --model resnet18 --split val --smoke-test --batch-size 32 --workers 0
```

本次正式实验运行的训练命令如下。两种模型都使用 seed 42、AdamW、weight decay 0.01、训练期 label smoothing 0.1、CosineAnnealingLR、CUDA AMP（若可用）、val top-1 模型选择及 patience 6。ResNet18 直接全网络微调，没有额外冻结阶段；学习率较低。train loss 含 label smoothing，val loss 不含，因此两条曲线适合分别观察趋势，数值不宜直接等同比较。

```powershell
.\.venv\Scripts\python.exe scripts\train.py --model simple-cnn --epochs 20 --batch-size 32 --workers 2 --patience 6
.\.venv\Scripts\python.exe scripts\train.py --model resnet18 --epochs 15 --batch-size 32 --workers 2 --patience 6
```

训练结束后，可先在完整 val 集合上核对独立评估流程。本次已运行该核对。最终 test 命令应在确定好模型与超参数后各执行一次：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --model simple-cnn --split val --batch-size 32 --workers 2
.\.venv\Scripts\python.exe scripts\evaluate.py --model resnet18 --split val --batch-size 32 --workers 2
.\.venv\Scripts\python.exe scripts\evaluate.py --model simple-cnn --split test --batch-size 32 --workers 2
.\.venv\Scripts\python.exe scripts\evaluate.py --model resnet18 --split test --batch-size 32 --workers 2
.\.venv\Scripts\python.exe scripts\build_report.py
```

重新执行 test 命令会再次评估并覆盖对应 JSON 与图表；不要用新的 test 数值反复选择模型。

## 本次真实运行结果

| 模型 | 最佳 epoch | 最佳 val top-1 | Test loss | Test top-1 | Test top-5 | Macro F1 | 参数量 | 训练时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 20 | 34.02% | 2.9476 | 29.27% | — | 26.59% | 415,110 | 76.1 秒 |
| ResNet18 | 15 | 90.10% | 0.9953 | 86.94% | 96.15% | 87.01% | 11,228,838 | 56.5 秒 |

两个模型正式训练合计 **132.7 秒**，均在 RTX 4070 SUPER 上运行。ResNet18 test macro precision 为 86.17%、macro recall 为 89.36%。结果来自 `outputs/metrics/*.json`；详细方法、曲线、混淆矩阵和典型预测分析见 [实验结果报告](reports/experiment-results.md)。只运行了一个固定种子，ResNet18 的训练准确率到达 100%，报告中说明了泛化差距及其他局限。

## 产物与课程提交

- `outputs/checkpoints/simple-cnn-best.pt`、`resnet18-best.pt`：按最佳 val accuracy 保存的 `state_dict` checkpoint，含优化器状态、epoch、配置、验证指标和环境信息。
- `outputs/metrics/`：数据检查、逐 epoch 历史、最终 test 指标 JSON。
- `outputs/figures/`：每个模型的 train/val loss、train/val accuracy、102 类混淆矩阵、含真值/预测/置信度的测试样例图，共 8 张 PNG。
- `outputs/validation/`、`outputs/smoke/`：验证流程和 smoke 产物，与最终指标分开。
- `reports/experiment-results.md`：从真实 JSON 生成的课程实验材料。

`data/`、`outputs/`、`.venv/` 均被 Git 忽略。提交课程作业时，需要按课程要求**另行附上**报告引用的图表、相关指标文件及可能要求的最佳模型权重；仅提交源码仓库不会包含这些生成物。
