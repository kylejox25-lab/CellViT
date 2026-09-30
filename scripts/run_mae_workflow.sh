#!/usr/bin/env bash
# Resume an existing MAE run and report its validation reconstruction quality.
# Activate the cellvit environment before running this script.
set -Eeuo pipefail

usage() {
  printf 'Usage: bash %s RUN_DIR [PREVIEW_BATCHES=16]\n' "$0"
  printf 'RUN_DIR must contain config.json and latest.pt. Existing epoch budget is preserved.\n'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 2 )) || [[ -z "${1:-${CELLVIT_RUN_DIR:-}}" ]]; then
  usage >&2
  exit 2
fi

RUN_DIR="${1:-${CELLVIT_RUN_DIR:-}}"
PREVIEW_BATCHES="${2:-16}"
if [[ ! "$PREVIEW_BATCHES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PREVIEW_BATCHES must be a positive integer.\n' >&2
  exit 2
fi
RUN_DIR="$(cd -- "$RUN_DIR" && pwd -P)"
for name in config.json latest.pt; do
  if [[ ! -f "$RUN_DIR/$name" ]]; then
    printf 'Missing file: %s/%s\n' "$RUN_DIR" "$name" >&2
    exit 1
  fi
done

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_DIR="$(mktemp -d "$RUN_DIR/workflow_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
ARGS_FILE="$(mktemp)"
trap 'rm -f -- "$ARGS_FILE"' EXIT
trap 'printf "Workflow stopped. See logs in: %s\n" "$REPORT_DIR" >&2' ERR

# Read config as data. NUL-separated arguments preserve spaces without eval.
"$PYTHON_BIN" - "$RUN_DIR" "$REPORT_DIR" "$ARGS_FILE" <<'PY'
import json
import sys
from pathlib import Path

run_dir, report_dir, args_file = map(Path, sys.argv[1:])
config = json.loads((run_dir / "config.json").read_text())
training = config["training"]
if Path(training["output"]).resolve() != run_dir:
    raise ValueError("Run directory differs from config.json; resume requires the original output path")
model_path = report_dir / "model_config.json"
model_path.write_text(json.dumps(config["model"], indent=2), encoding="utf-8")
arguments = [training["mds_root"]]
for key, value in training.items():
    if key == "amp":
        if not value:
            arguments.append("--no-amp")
    else:
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
arguments.extend(["--model-config", str(model_path), "--resume", str(run_dir / "latest.pt")])
args_file.write_bytes(b"\0".join(arg.encode() for arg in arguments) + b"\0")
print(f"Run: {run_dir}\nReports: {report_dir}")
print(f"Total epochs: {training['epochs']}; batch: {training['batch_size']}; "
      f"accumulation: {training['accumulation_steps']}; AMP: {training['amp']}")
PY

ARGS=()
while IFS= read -r -d '' argument; do
  ARGS+=("$argument")
done < "$ARGS_FILE"
MDS_ROOT="${ARGS[0]}"
TRAIN_ARGS=("${ARGS[@]:1}")

printf '\n[1/3] Evaluate the current checkpoint on %s batches\n' "$PREVIEW_BATCHES"
"$PYTHON_BIN" -u -m cellvit.evaluate \
  --checkpoint "$RUN_DIR/latest.pt" \
  --mds-root "$MDS_ROOT" --split val \
  --max-batches "$PREVIEW_BATCHES" --log-every 4 \
  --output "$REPORT_DIR/preview.json" \
  2>&1 | tee "$REPORT_DIR/preview.log"

printf '\n[2/3] Resume training to the original epoch budget\n'
# No --max-steps: finish the remaining epochs. All saved settings are preserved.
"$PYTHON_BIN" -u -m cellvit.train.mae "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$REPORT_DIR/train.log"

printf '\n[3/3] Report the best reconstruction checkpoint\n'
CHECKPOINT="$RUN_DIR/best_reconstruction.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/latest.pt"
fi

# Reuse the full validation already performed at epoch end. Older checkpoints
# without saved metrics fall back to an independent full evaluation below.
"$PYTHON_BIN" - "$CHECKPOINT" "$MDS_ROOT" "$REPORT_DIR/validation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint, root, output = map(Path, sys.argv[1:])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
content = (root / "manifest.json").read_bytes()
if state["manifest_sha256"] != hashlib.sha256(content).hexdigest():
    raise ValueError("Checkpoint and MDS manifest do not match")
expected = json.loads(content)["counts"]["val"]
metrics = state.get("validation_metrics")
if metrics and metrics["wells"] == expected and metrics.get("max_batches") is None:
    report = {
        **metrics, "checkpoint": str(checkpoint), "split": "val",
        "step": state["global_step"], "batch_size": state["run_config"]["batch_size"],
        "expected_wells": expected, "complete_split": True,
        "manifest_sha256": state["manifest_sha256"],
        "report_source": "checkpoint_validation",
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Reused complete validation metrics from:", checkpoint)
    print("Masked MSE:", report["masked_mse"])
    print("Reconstruction skill:", report["reconstruction_skill"])
else:
    print("No complete saved validation metrics; running full evaluation next.")
PY

if [[ ! -f "$REPORT_DIR/validation.json" ]]; then
  "$PYTHON_BIN" -u -m cellvit.evaluate \
    --checkpoint "$CHECKPOINT" \
    --mds-root "$MDS_ROOT" --split val --log-every 50 \
    --output "$REPORT_DIR/validation.json" \
    2>&1 | tee "$REPORT_DIR/validation.log"
fi

printf '\nDone. Validation report: %s/validation.json\n' "$REPORT_DIR"
printf 'For plots, run:\n'
printf '  export CELLVIT_RUN_DIR=%q\n' "$RUN_DIR"
printf '  jupyter lab notebooks/02_mae_training_and_reconstruction.ipynb\n'
