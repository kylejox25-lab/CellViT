# CellViT

当前阶段仅实现数据读取。原始 RxRx3-core 目录保持只读；转换后的 MDS、缓存与日志放在新的可写目录。

## 数据流

`35 个原始 Parquet 分片 → 每孔 6 个 JP2 字节 → MDS train/val/test → StreamingDataLoader`

转换器保留 JP2 压缩字节，不把图像预先展开成大数组；每个 MDS 样本对应一个孔位。读取器解码后返回 `image`：`[batch, 4, 6, 256, 256]`，数值为 `float32 [0,1]`。四块裁剪顺序为左上、右上、左下、右下。训练阶段选择其中一块的策略留给训练循环决定，以保证断点续训时不产生隐藏的随机裁剪状态。

数据按每个实验内的整块孔板分成 train/val/test。`manifest.json` 记录数量、源分片大小、元数据哈希和转换是否完整；`plate_splits.json` 固定划分。`--max-wells` 只用于小样本冒烟测试，产生的 `complete=false` 数据不能用于正式训练。
常规 `make_dataloader` 会拒绝读取不完整转换；检查命令仅为冒烟测试允许读取。

## Conda 环境

在服务器的项目目录中直接创建环境（目标为单张 RTX 3090；PyTorch 2.5.1 + CUDA 12.1；MKL 固定为兼容版本）：

```bash
conda env create -f environment.yml
conda activate cellvit
python -m pip install -e . --no-deps
python -c "import torch, torchvision, streaming; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), streaming.__version__)"
python -m pytest -q
```

如果环境已经存在，先运行 `conda env update -f environment.yml --prune`，再执行安装项目的命令。`torch.cuda.is_available()` 在 3090 服务器上应为 `True`；如果为 `False`，先检查驱动和服务器的 CUDA 环境，不要开始训练。

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
# torch.Size([8, 4, 6, 256, 256])
```

`make_dataloader` 使用 MosaicML 的 `StreamingDataLoader`，训练检查点应同时保存其 `state_dict()`；恢复时调用 `load_state_dict()`。DataLoader 与 StreamingDataset 使用相同的每卡 batch size。

## 依据

- [MosaicML Streaming 快速入门](https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/quick_start.html)
- [MDS 转换和字节列](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
- [StreamingDataset 与 batch_size](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html)
- [StreamingDataLoader 的恢复接口](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataLoader.html)
