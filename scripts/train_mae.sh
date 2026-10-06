#!/bin/bash
set -e

# 在项目根目录运行，先 conda activate cellvit。
MDS_ROOT="/NFS2_home/NFS2_home_3/xkj2006/cell_painting/project/data/rxrx3_mds"  # 改成你的 MDS 数据目录
RUN_DIR="/NFS2_home/NFS2_home_3/xkj2006/cell_painting/project/runs/mae_batch2_50epochs"
EPOCHS=50  # 起始实验预算，不代表一定收敛；首次训练的 RUN_DIR 必须不存在。

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

cellvit-evaluate \
  --checkpoint "$RUN_DIR/best_reconstruction.pt" \
  --mds-root "$MDS_ROOT" \
  --split val \
  --batch-size 2 \
  --num-workers 4 \
  --log-every 50 \
  --output "$RUN_DIR/validation_report.json"

# 3. 查看重建指标。
cat "$RUN_DIR/validation_report.json"