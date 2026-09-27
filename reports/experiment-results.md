# Oxford Flowers102 CNN 分类实验结果

## 1. 实验题目与目的

基于卷积神经网络完成 Oxford Flowers102 **全部 102 类**细粒度花卉分类。比较自行实现的 SimpleCNN 基线和采用 ImageNet 预训练权重的 ResNet18，形成可复现的训练、验证、最终测试及课程报告材料。

## 2. 数据集及官方划分

数据来自 [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)，通过 [torchvision.datasets.Flowers102](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html) 下载和读取。官方 `setid.mat` 决定划分；未重新随机切分，也未挑选 38 类子集。数据完整性检查确认三个 split 的图像 ID 互不重叠，每个 split 覆盖 102 类，标签为 0–101。

| split | 样本数 | 用途 |
| --- | ---: | --- |
| train | 1020 | 参数训练 |
| val | 1020 | checkpoint 选择、停止判断 |
| test | 6149 | 训练结束后最终评估 |

总计 8189 张。数据抽样读取验证了 `float32`、`3×224×224` 张量。

## 3. 预处理

训练图像使用 `RandomResizedCrop(224, scale=(0.7, 1.0))`、随机水平翻转、`RandomRotation(10)`、`ToTensor`；val/test 使用 `Resize(256)`、`CenterCrop(224)`、`ToTensor`。三者均使用 ImageNet 预训练权重要求的均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 归一化。训练增强未用于 val/test。

## 4. CNN 原理与模型结构

CNN 用可学习卷积核提取局部图像特征，逐层组合为更高层的形状与纹理表征；池化压缩空间尺寸，最终分类层给出 102 类 logits。

**SimpleCNN：**四个 `Conv2d → BatchNorm2d → ReLU → MaxPool2d` 模块，通道数为 32、64、128、256；随后 `AdaptiveAvgPool2d(1) → Flatten → Dropout(0.3) → Linear(102)`。参数量 415,110，全部可训练。

**ResNet18：**使用 torchvision 官方 `ResNet18_Weights.IMAGENET1K_V1`，将最后全连接层替换成 102 类输出。残差连接让网络更容易训练。采用直接全网络微调，学习率低于 SimpleCNN；参数量 11,228,838，全部可训练。预训练权重保存在项目 `data/torch-hub/`，没有使用 test 数据训练。

## 5. 实验环境与超参数

- Python 3.12.3；PyTorch 2.13.0+cu130；torchvision 0.28.0+cu130；NumPy 2.5.2。
- 设备：NVIDIA GeForce RTX 4070 SUPER；PyTorch CUDA 构建 13.0；CUDA AMP 已启用。
- 随机种子 42，固定 Python、NumPy、PyTorch 和 CUDA 随机源；cuDNN 使用 deterministic 模式。
- 优化器 AdamW；损失函数为训练期交叉熵加 label smoothing，评估期为普通交叉熵；学习率调度为单个 CosineAnnealingLR。以 val top-1 accuracy 保存最佳 checkpoint，连续 6 个 epoch 未改进时提前停止。
- 因训练与验证损失的 label smoothing 设置不同，两条 loss 曲线可看各自变化趋势，但数值不宜直接等同比较。

| 模型 | 最大 epoch | batch | 初始学习率 | weight decay | label smoothing | patience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 20 | 32 | 1e-03 | 0.01 | 0.1 | 6 |
| ResNet18 | 15 | 32 | 1e-04 | 0.01 | 0.1 | 6 |

## 6. 训练过程与结果

正式训练前，两个模型均通过少量真实 batch 的 smoke test：数据加载、CUDA forward/backward、validation、checkpoint 保存/加载以及评估 JSON 均成功。Smoke 输出与正式结果隔离在 `outputs/smoke/`。

| 模型 | 最佳 epoch | 最佳 val top-1 | Test loss | Test top-1 | Test top-5 | Macro precision | Macro recall | Macro F1 | 参数量 | 训练秒数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 20 | 34.02% | 2.9476 | 29.27% | — | 28.65% | 33.75% | 26.59% | 415,110 | 76.1 |
| ResNet18 | 15 | 90.10% | 0.9953 | 86.94% | 96.15% | 86.17% | 89.36% | 87.01% | 11,228,838 | 56.5 |

SimpleCNN 最佳 epoch 的 train top-1 为 46.57%；ResNet18 为 100.00%。两次正式训练合计 132.7 秒（不含数据下载、smoke、绘图和最终测试）。两个模型均运行至设定最大 epoch，未触发 early stopping。最佳模型分别保存在 `outputs/checkpoints/simple-cnn-best.pt` 与 `outputs/checkpoints/resnet18-best.pt`。

ResNet18 的 test top-1 比 SimpleCNN 高 57.67 个百分点。该对比只代表本次固定划分、配置和随机种子的结果；ResNet18 借助 ImageNet 预训练，不是与从零训练基线完全相同的起点。

## 7. 混淆矩阵与典型样例

ResNet18 在 test 上的 102×102 混淆矩阵以类别编号显示。对角线集中，仍有以下较多的特定误分类（单元格计数）：

- 真值 45（wallflower）→ 预测 88（watercress）：20 张。
- 真值 73（rose）→ 预测 87（cyclamen）：16 张。
- 真值 50（petunia）→ 预测 31（garden phlox）：16 张。
- 真值 72（water lily）→ 预测 77（lotus）：15 张。
- 真值 50（petunia）→ 预测 85（tree mallow）：14 张。

这些是观察到的预测模式，不能仅凭计数断言混淆的成因。示例图同时给出真值、预测和模型置信度，包含正确与错误样本：

- 样本索引 4958：真值 columbine，预测 windflower，预测置信度 21.11%，错误。
- 样本索引 1081：真值 sweet william，预测 great masterwort，预测置信度 7.24%，错误。
- 样本索引 2075：真值 marigold，预测 marigold，预测置信度 96.24%，正确。
- 样本索引 1284：真值 ruby-lipped cattleya，预测 ruby-lipped cattleya，预测置信度 13.09%，正确。

![ResNet18 测试集混淆矩阵](../outputs/figures/resnet18-confusion-matrix.png)

![ResNet18 测试样本预测](../outputs/figures/resnet18-sample-predictions.png)

## 8. 图表与机器可读结果

- SimpleCNN：[loss 曲线](../outputs/figures/simple-cnn-training-loss.png)、[accuracy 曲线](../outputs/figures/simple-cnn-training-accuracy.png)、[混淆矩阵](../outputs/figures/simple-cnn-confusion-matrix.png)、[预测样例](../outputs/figures/simple-cnn-sample-predictions.png)。
- ResNet18：[loss 曲线](../outputs/figures/resnet18-training-loss.png)、[accuracy 曲线](../outputs/figures/resnet18-training-accuracy.png)、[混淆矩阵](../outputs/figures/resnet18-confusion-matrix.png)、[预测样例](../outputs/figures/resnet18-sample-predictions.png)。
- JSON：`outputs/metrics/dataset.json`、`simple-cnn-history.json`、`resnet18-history.json`、`simple-cnn.json`、`resnet18.json`。

## 9. 结论与局限

本次实验完成了官方 102 类数据上的两种 CNN 训练和独立测试。ResNet18 获得 86.94% test top-1 与 87.01% macro F1；从零训练的 SimpleCNN 为 29.27% test top-1。预训练表征在每类仅 10 张训练图的条件下带来显著优势。

局限：只有一次固定种子的训练，没有多次运行的方差估计；ResNet18 训练准确率达到 100.00%，而 val/test 更低，显示泛化差距；类别测试样本数不均衡，单一总体准确率不能替代宏平均指标。测试集仅用于本次最终评估，未根据测试表现重新调参。课程提交时应将被 Git 忽略的图表和所需指标文件与报告一起交付。
