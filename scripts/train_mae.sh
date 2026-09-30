#!/usr/bin/env bash
set -e

# 在项目根目录运行，先 conda activate cellvit。
MDS_ROOT="/path/to/rxrx3_mds"  # 改成你的完整 MDS 数据目录
RUN_DIR="/NFS2_home/NFS2_home_3/xkj2006/cell_painting/project/runs/mae_batch2_20epochs"
EPOCHS=20  # 起始实验预算，不代表一定收敛；首次训练的 RUN_DIR 必须不存在。

# 每轮结束自动验证，保存 latest.pt 和 best_reconstruction.pt。
# 中断后在命令末尾追加 --resume "$RUN_DIR/latest.pt"，其它参数保持一致。
cellvit-train \
  --mds-root "$MDS_ROOT" \
  --output "$RUN_DIR" \
  --model-config configs/mae_small.json \
  --epochs "$EPOCHS" \
  --batch-size 2 \
  --accumulation-steps 32 \
  --checkpoint-every 500 \
  --log-every 50
