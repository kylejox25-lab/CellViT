#!/bin/bash
set -e

# 在项目根目录运行，先 conda activate cellvit。
MDS_ROOT="/NFS2_home/NFS2_home_3/xkj2006/cell_painting/project/data/rxrx3_mds"  # 改成你的 MDS 数据目录
RUN_DIR="/NFS2_home/NFS2_home_3/xkj2006/cell_painting/project/runs/mae_batch2_test"

# 1. 用 16 个 batch 测试当前模型。
cellvit-evaluate \
  --checkpoint "$RUN_DIR/latest.pt" \
  --mds-root "$MDS_ROOT" \
  --split val \
  --max-batches 16 \
  --log-every 4 \
  --output "$RUN_DIR/preview_$(date +%Y%m%d_%H%M%S).json"

# 2. 继续完成原定的 1 个 epoch；结束后自动完整验证并保存模型。
# 这些参数与之前 batch2 测试的配置保持一致。
cellvit-train \
  --mds-root "$MDS_ROOT" \
  --output "$RUN_DIR" \
  --model-config configs/mae_small.json \
  --epochs 1 \
  --batch-size 2 \
  --accumulation-steps 32 \
  --checkpoint-every 5 \
  --log-every 1 \
  --resume "$RUN_DIR/latest.pt"

# 完整验证结果在 metrics.jsonl；绘图使用 02 Notebook。
