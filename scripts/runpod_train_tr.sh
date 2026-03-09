#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script currently supports Debian/Ubuntu-based RunPod images only." >&2
  exit 1
fi

LANG_CODE="${LANG_CODE:-tr}"
RUN_NAME="${RUN_NAME:-${LANG_CODE}-$(date +%Y%m%d-%H%M%S)}"
VENV_DIR="${VENV_DIR:-.venv-runpod}"
NUM_SHARDS="${NUM_SHARDS:-10}"
N_WORKERS="${N_WORKERS:--1}"
BATCH_SIZE="${BATCH_SIZE:-192}"
NUM_STEPS="${NUM_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
BASE_CONFIG="${BASE_CONFIG:-Configs/config_ml.yml}"

RUN_DIR="${RUN_DIR:-runs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/multilingual-phonemes.processed}"
SHARD_DIR="${SHARD_DIR:-${RUN_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/checkpoints}"
GENERATED_CONFIG="${GENERATED_CONFIG:-${RUN_DIR}/config_ml.yml}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"

mkdir -p "$RUN_DIR" "$HF_HOME"

echo "[1/6] Installing system packages"
$SUDO apt-get update
$SUDO apt-get install -y git espeak-ng python3-venv

echo "[2/6] Creating Python environment at ${VENV_DIR}"
python3 -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[3/6] Installing Python packages"
if ! python -c "import torch" >/dev/null 2>&1; then
  pip install torch torchvision torchaudio
fi
pip install \
  pandas \
  singleton-decorator \
  datasets \
  "transformers<4.33.3" \
  accelerate \
  nltk \
  phonemizer \
  sacremoses \
  pebble \
  pyyaml \
  tqdm \
  tensorboard

echo "[4/6] Generating run config at ${GENERATED_CONFIG}"
python - "$BASE_CONFIG" "$GENERATED_CONFIG" "$DATA_DIR" "$LOG_DIR" "$BATCH_SIZE" "$NUM_STEPS" "$SAVE_INTERVAL" "$LOG_INTERVAL" "$MIXED_PRECISION" <<'PY'
import sys
import yaml

base_config, output_config, data_dir, log_dir, batch_size, num_steps, save_interval, log_interval, mixed_precision = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["data_folder"] = data_dir
config["log_dir"] = log_dir
config["batch_size"] = int(batch_size)
config["num_steps"] = int(num_steps)
config["save_interval"] = int(save_interval)
config["log_interval"] = int(log_interval)
config["mixed_precision"] = mixed_precision

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "[5/6] Preprocessing ${LANG_CODE} dataset"
python preprocess_ml.py \
  --lang "$LANG_CODE" \
  --config_path "$GENERATED_CONFIG" \
  --root_directory "$SHARD_DIR" \
  --num_shards "$NUM_SHARDS" \
  --n_workers "$N_WORKERS"

echo "[6/6] Starting training"
python train.py --config_path "$GENERATED_CONFIG"
