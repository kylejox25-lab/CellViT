# CellViT

当前已实现数据读取、空间掩码 MAE 训练、遮挡重建评估与孔位 embedding 导出。训练曲线和六通道重建图见 [02 重建评估 Notebook](notebooks/02_mae_training_and_reconstruction.ipynb)。原始 RxRx3-core 目录保持只读；转换后的 MDS、缓存与日志放在新的可写目录。模型效果需要在服务器实际运行后测定。

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

验证重建损失用于评价 MAE 本身的遮挡重建任务；最低验证 MSE 对应的完整检查点另存为 `best_reconstruction.pt`。这个名字只代表重建最佳。跨 guide、跨板检索评测尚未实现。

### AMP 溢出与训练进度

梯度累积先按计划的有效 batch 大小缩小 loss，再反向传播；在更新前按实际孔位数校正，最后不足一组的数据同样得到正确平均梯度。AMP 梯度出现 Inf/NaN 时，先由 `unscale_` 记录，再让 GradScaler 跳过本次参数更新、降低缩放系数，最后清空梯度；不对这些非有限梯度做裁剪。FP32 梯度异常、非有限 loss 或有限梯度的范数计算溢出仍会明确报错。

`step` 只计成功更新；`update_attempts` 包含 AMP 跳过。学习率、`--log-every` 和 `--checkpoint-every` 按更新尝试次数推进，确保每个 epoch 的计划进度不因跳过而延后；`--max-steps` 仍限制成功更新总数。日志新增 `amp_overflow`、`skipped_updates`、`grad_norm`、`loss_scale_before`、`loss_scale`。连续 20 次溢出会保存 `latest.pt` 后停止，防止长期没有实际更新。

原有 checkpoint 仍可用相同命令续训，缺少的新计数字段会自动补默认值；旧 checkpoint 的“最佳重建”从恢复后的验证开始记录。已完成全部 epoch 的旧 checkpoint 可以直接用下面的评估命令，无需重新训练。

轮末保存和恢复时，将 Streaming 游标显式对齐为下一轮 `epoch`、`sample_in_epoch=0`；旧 checkpoint 也会在读取时修正。中途保存的样本偏移保持原值。该处理依据 [Streaming 0.13.0 的游标实现](https://github.com/mosaicml/streaming/blob/v0.13.0/streaming/base/dataset.py)，避免训练计数与数据读取位置落在不同 epoch。

## 评价 MAE 的遮挡重建质量

### 多轮正式训练

`scripts/train_mae.sh` 从头启动多轮训练，默认 20 个 epoch、batch size 2、梯度累积 32。修改脚本顶部的 MDS 路径和新输出目录后，在项目根目录执行：

```bash
conda activate cellvit
bash scripts/train_mae.sh
```

每轮自动完整验证并保存最佳重建模型。20 轮是起始实验预算，是否训练充分应根据验证 MSE 和相对重建评分的趋势判断；当前不自动早停。中断时在训练命令末尾追加 `--resume "$RUN_DIR/latest.pt"`，总轮数及其余训练参数保持不变。之前 1 轮测试任务的 checkpoint 不能直接改为 20 轮续训。

### 续训和评估脚本

修改 `scripts/run_mae_workflow.sh` 顶部的 `MDS_ROOT` 和 `RUN_DIR`，然后在服务器项目根目录执行：

```bash
conda activate cellvit
bash scripts/run_mae_workflow.sh
```

脚本先用 16 个 batch 评价 `latest.pt`，再按当前 `batch-size=2`、`accumulation-steps=32` 的配置继续完成原定 1 个 epoch。参数直接写在脚本中，应与已有 checkpoint 保持一致。

预览报告保存为 `RUN_DIR/preview_时间.json`。训练在轮末自动完整验证，结果写入 `metrics.jsonl`，绘图使用 `02_mae_training_and_reconstruction.ipynb`。

这是连续像素回归，不报告词/token 分类准确率。训练与评估共享 `model.reconstruction_target()`，默认目标是每个 patch 内每个通道的标准化像素。所有指标只统计 **被遮挡的位置**。

| 字段 | 定义与解释 |
| --- | --- |
| `masked_mse` | 隐藏像素的平均平方误差，越低越好；与训练目标一致 |
| `masked_rmse` | MSE 的平方根，单位为目标空间单位 |
| `zero_baseline_mse` | 相同隐藏位置全部预测为 0 的误差，实际计算，不假设它等于 1 |
| `reconstruction_skill` | `1 - masked_mse / zero_baseline_mse`，1 表示完美、0 表示与基线相同、负值表示更差；0.4 表示误差降低 40%，不是准确率 40% |
| `per_channel_*` | 分别记录六个通道的 MSE、基线及相对评分 |

评分先累计所有隐藏位置的误差，再计算比值；不平均每个 batch 的百分比。基线能量接近零时，相对评分记为 `null`，MSE 仍有效。关闭 `norm_pix_loss` 时目标变为原始 `[0,1]` 像素，零基线也随之变成黑色像素基线，这两种设置不能混比。

轮末验证自动将指标以 `val_` 前缀写入 `metrics.jsonl`，保留旧的 `val_reconstruction_loss` 字段。还可随时评价已有 `latest.pt` 或 `best_reconstruction.pt`：

```bash
# 更新入口命令；使用已配置的 cellvit 环境，不重装依赖。
python -m pip install -e . --no-deps

cellvit-evaluate --checkpoint /path/to/runs/mae_smoke/latest.pt \
  --mds-root /path/to/rxrx3_mds --split val \
  --output /path/to/runs/mae_smoke/reconstruction_val.json
```

默认遍历整个 split，沿用 checkpoint 的 batch size、训练 seed + 10000 和 AMP 配置；没有 GPU 时使用 FP32 CPU。`--max-batches 32` 可做部分评估，报告包含 `wells`、`expected_wells`、`complete_split`，不能将部分结果当作整份验证集成绩。`--no-amp` 可以辅助排查推理溢出；精度会记录在报告里。输出文件需不存在。比较模型时固定数据、目标标准化、batch size、遮挡率、种子、样本顺序和计算精度。

评估现在显示 checkpoint 加载、数据准备和运行进度：默认每 50 个 batch 或完成一个 batch 后距上次报告达到 30 秒时，输出当前 MSE、孔位数、速度、耗时和 ETA。`--log-every 10` 可提高频率，`--log-every 0` 关闭进度。进度写入 stderr，最终 JSON 保留在 stdout。报告新增 `elapsed_seconds`、`wells_per_second` 和 `loader_wait_seconds`；后者是主进程等待 DataLoader 返回样本的时间，不是 worker 解码时间的总和。模型前向返回的目标直接用于评估，避免重复标准化；通道误差在设备上累计，报告时才复制到 CPU。实际提速需要服务器实测。

绘图只放在 Notebook 中：

```bash
export CELLVIT_RUN_DIR=/path/to/runs/mae_smoke
jupyter lab notebooks/02_mae_training_and_reconstruction.ipynb
```

Notebook 可读取旧/新日志，显示训练与验证曲线、AMP 缩放记录，直接评估 checkpoint 并绘制六通道指标与重建图。重建展示在训练目标空间中进行，不借用隐藏 patch 的真实均值/方差恢复原始强度。显示时可见区域复制真实目标，误差仍只统计隐藏区域。默认 `MAX_BATCHES=None` 为全量评估，也可改为明确标注的部分预览。

## 导出孔位 embedding

使用训练检查点为一个 split 中的每孔输出一个未归一化向量：

```bash
cellvit-embed --checkpoint /path/to/runs/mae_smoke/latest.pt \
  --mds-root /path/to/rxrx3_mds --split val \
  --output /path/to/embeddings/mae_val.parquet --batch-size 1
```

导出器检查向量有限、`well_id` 无重复、总孔数与 MDS 清单一致，并核对检查点的数据清单哈希。输出 Parquet 包含 `well_id` 和固定长度 `embedding` 列；后续评测可按 `well_id` 连接原始元数据。转换、训练和导出均不会修改原始 RxRx3-core 图像目录。

导出时沿用 checkpoint 的 AMP 设置；FP32 训练的模型不会在导出阶段自动切成 FP16。输出路径不得位于 MDS 数据目录内。

## 依据

- [MosaicML Streaming 快速入门](https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/quick_start.html)
- [MDS 转换和字节列](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
- [StreamingDataset 与 batch_size](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html)
- [StreamingDataLoader 的恢复接口](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataLoader.html)
