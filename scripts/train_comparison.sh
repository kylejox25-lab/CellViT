#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, after activating the existing cellvit environment.
# All four runs use the same images, plate split, seed, and effective batch.
MDS_ROOT="${RXRX3_MDS_ROOT:?Set RXRX3_MDS_ROOT to a complete MDS conversion}"
RUN_ROOT="${CELLVIT_COMPARISON_ROOT:-runs/comparison_patch8_50epochs}"
BATCH_SIZE="${CELLVIT_BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${CELLVIT_ACCUMULATION_STEPS:-64}"

for MODEL in mae ca_mae diffmae cnn_mae; do
  CONFIG="configs/${MODEL}.json"
  if [[ "$MODEL" == "mae" ]]; then CONFIG="configs/mae_comparison.json"; fi
  RUN_DIR="$RUN_ROOT/$MODEL"
  RESUME_ARGS=()
  if [[ "${CELLVIT_RESUME:-0}" == "1" && -f "$RUN_DIR/latest.pt" ]]; then
    RESUME_ARGS=(--resume "$RUN_DIR/latest.pt")
  fi
  cellvit-train --mds-root "$MDS_ROOT" --output "$RUN_DIR" \
    --model-config "$CONFIG" --epochs 50 --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" --seed 2026 \
    --checkpoint-every 500 --log-every 50 "${RESUME_ARGS[@]}"

  # Outputs must not exist. A restarted script preserves completed exports.
  for SPLIT in val test; do
    OUTPUT="$RUN_DIR/embedding_${SPLIT}.parquet"
    if [[ ! -f "$OUTPUT" ]]; then
      cellvit-embed --checkpoint "$RUN_DIR/best_reconstruction.pt" \
        --mds-root "$MDS_ROOT" --split "$SPLIT" --output "$OUTPUT" \
        --batch-size "$BATCH_SIZE"
    fi
  done
done
