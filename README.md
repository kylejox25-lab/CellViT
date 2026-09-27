# CellViT

当前已实现数据读取、空间掩码 MAE 基线的训练与孔位 embedding 导出。原始 RxRx3-core 目录保持只读；转换后的 MDS、缓存与日志放在新的可写目录。服务器上的完整训练和生物学检索评测仍需执行。

## 数据流

`35 个原始 Parquet 分片 → 每孔 6 个 JP2 字节 → MDS train/val/test → StreamingDataLoader`

转换器保留 JP2 压缩字节，不把图像预先展开成大数组；每个 MDS 样本对应一个孔位。读取器解码后返回完整孔位图像 `image`：`[batch, 6, 512, 512]`，数值为 `float32 [0,1]`。不在读取阶段裁切。模型以 16×16 图像 patch 处理完整孔位，每孔对应 1024 个空间 token；尺寸定义见 `src/cellvit/image_config.py`。

数据按每个实验内的整块孔板分成 train/val/test。`manifest.json` 记录数量、源分片大小、元数据哈希和转换是否完整；`plate_splits.json` 固定划分。`--max-wells` 只用于小样本冒烟测试，产生的 `complete=false` 数据不能用于正式训练。
常规 `make_dataloader` 会拒绝读取不完整转换；检查命令仅为冒烟测试允许读取。

## Conda 环境

在服务器的项目目录中直接创建环境（目标为单张 RTX 3090；PyTorch 2.5.1 + CUDA 12.1；MKL 固定为兼容版本）：

```bash
conda env create -f environment.yml
conda activate cellvit
python -m pip install -e . --no-deps
python -c "import torch, torchvision, streaming; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available(), streaming.__version__)"
python -m pytest -q
```

如果环境已经存在，先运行 `conda env update -f environment.yml --prune`，再执行安装项目的命令。环境应使用 `pytorch 2.5.1` 的 CUDA 12.1 构建和 `torchvision 0.20.1` 的 cu121 构建；CPU 版 PyTorch 与 cu121 版 torchvision 混装会在导入时失败。`torch.version.cuda` 应为 `12.1`，`torch.cuda.is_available()` 在 3090 服务器上应为 `True`。

要在 Jupyter 中使用该环境：

```bash
python -m ipykernel install --user --name cellvit --display-name "Python (cellvit)"
jupyter lab
```

打开 [数据检查 notebook](notebooks/01_explore_rxrx3_core.ipynb)，选择 `Python (cellvit)` 内核。Notebook 默认读取本机的 `E:\CellPainting\rxrx3_core`；在服务器上设置 `RXRX3_CORE_ROOT` 环境变量，或直接修改 notebook 第一段配置中的路径。

Linux 服务器示例：

```bash
export RXRX3_CORE_ROOT=/path/to/rxrx3_core
jupyter lab notebooks/01_explore_rxrx3_core.ipynb
```

如需查看已转换的 MDS，再设置 `RXRX3_MDS_ROOT=/path/to/rxrx3_mds` 并重新运行 notebook。准备提交 GitHub 时清空 notebook 的运行输出，避免把数据集图像预览嵌入提交。

## 数据转换与读取

先做小样本转换（源目录和输出目录替换为服务器实际路径）：

```bash
cellvit-convert --source /path/to/rxrx3_core --output /path/to/rxrx3_smoke --max-wells 12
```

检查冒烟结果后，转换全部数据至另一个**不存在**的输出目录：

```bash
cellvit-convert --source /path/to/rxrx3_core --output /path/to/rxrx3_mds
```

先查看小样本 `manifest.json` 中哪个 split 有样本，再对该 split 运行真实 MDS 读取检查：

```bash
cellvit-check-loader --mds-root /path/to/rxrx3_smoke --split test --batch-size 2
```

上面的 `test` 只是示例；小样本的前几个孔位可能属于任意一个 split。

正式转换会再占用接近原始图像数据体量的磁盘空间。转换中断时保留 `<output>.incomplete` 供检查；转换器不会删除或覆盖已有目录。

读取示例：

```python
from cellvit.data.streaming_dataset import make_dataloader

loader = make_dataloader(mds_root="/path/to/rxrx3_mds", split="train", batch_size=8)
batch = next(iter(loader))
print(batch["image"].shape, batch["well_id"][:2])
# torch.Size([8, 6, 512, 512])
```

`make_dataloader` 使用 MosaicML 的 `StreamingDataLoader`，训练检查点应同时保存其 `state_dict()`；恢复时调用 `load_state_dict()`。DataLoader 与 StreamingDataset 使用相同的每卡 batch size。

## 空间 MAE 基线训练

`src/cellvit/models/mae.py` 是从零初始化的 ViT Small/16 编码器和轻量解码器。训练时随机遮住 75% 的空间 patch，编码器只处理其余 patch，解码器重建被遮住的像素。每个 patch 内按通道标准化重建目标，再对被遮住的位置求平均损失。推理时只用编码器，将所有 patch token 平均池化成一个孔位向量。默认架构见 [configs/mae_small.json](configs/mae_small.json)。

在已完成的 MDS 上先运行很短的服务器冒烟训练；输出目录必须尚不存在：

```bash
cellvit-train --mds-root /path/to/rxrx3_mds --output /path/to/runs/mae_smoke \
  --model-config configs/mae_small.json --epochs 1 --batch-size 1 \
  --accumulation-steps 64 --checkpoint-every 5 --max-steps 10
```

该命令处理完整孔位图像，不接受 `complete=false` 的小样本 MDS。`--max-steps` 是**累计优化器步数上限**，适合检查前向、反向、显存和恢复流程；它并不表示完成一个 epoch。接着用相同配置从 `latest.pt` 恢复，省略 `--max-steps` 可继续完成原定 epoch：

```bash
cellvit-train --mds-root /path/to/rxrx3_mds --output /path/to/runs/mae_smoke \
  --model-config configs/mae_small.json --epochs 1 --batch-size 1 \
  --accumulation-steps 64 --checkpoint-every 5 \
  --resume /path/to/runs/mae_smoke/latest.pt
```

正式实验改用新的输出目录和预定的 epoch 数。单张 3090 从 `batch-size 1` 起核查显存，再决定是否增大；梯度累积在优化器更新前按实际孔位数平均梯度。训练使用 FP16 自动混合精度、梯度缩放、AdamW、预热加余弦学习率；默认学习率为 `1.5e-4`。如需取消混合精度，可传 `--no-amp`。`config.json` 保存运行配置和 MDS 清单哈希，`metrics.jsonl` 记录训练损失、学习率、吞吐、峰值显存和每轮验证重建损失，`latest.pt` 保存模型、优化器、缩放器、读取进度和随机状态。恢复时需保持原训练参数与数据清单一致。

验证重建损失只用于监控训练；它不是最终模型选择指标。跨 guide、跨板检索评测尚未实现，不应以像素重建损失宣称 embedding 的生物学质量。

## 导出孔位 embedding

使用训练检查点为一个 split 中的每孔输出一个未归一化向量：

```bash
cellvit-embed --checkpoint /path/to/runs/mae_smoke/latest.pt \
  --mds-root /path/to/rxrx3_mds --split val \
  --output /path/to/embeddings/mae_val.parquet --batch-size 1
```

导出器检查向量有限、`well_id` 无重复、总孔数与 MDS 清单一致，并核对检查点的数据清单哈希。输出 Parquet 包含 `well_id` 和固定长度 `embedding` 列；后续评测可按 `well_id` 连接原始元数据。转换、训练和导出均不会修改原始 RxRx3-core 图像目录。

## 依据

- [MosaicML Streaming 快速入门](https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/quick_start.html)
- [MDS 转换和字节列](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
- [StreamingDataset 与 batch_size](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html)
- [StreamingDataLoader 的恢复接口](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataLoader.html)
