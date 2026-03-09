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
RUN_NAME="${RUN_NAME:-turkish-h100}"
VENV_DIR="${VENV_DIR:-.venv-runpod}"
NUM_SHARDS="${NUM_SHARDS:-10}"
N_WORKERS="${N_WORKERS:--1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_STEPS="${NUM_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAX_MEL_LENGTH="${MAX_MEL_LENGTH:-384}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
BASE_CONFIG="${BASE_CONFIG:-Configs/config_ml.yml}"

RUN_DIR="${RUN_DIR:-runs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/multilingual-phonemes.processed}"
SHARD_DIR="${SHARD_DIR:-${RUN_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/checkpoints}"
GENERATED_CONFIG="${GENERATED_CONFIG:-${RUN_DIR}/config_ml.yml}"
CONDA_DIR="${CONDA_DIR:-${REPO_ROOT}/.miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-plbert-py311}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN_DIR" "$HF_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=1
fi
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
fi

if [[ -z "$BATCH_SIZE" ]]; then
  BATCH_SIZE="$((PER_DEVICE_BATCH_SIZE * GPU_COUNT))"
fi

echo "[1/6] Installing system packages"
$SUDO apt-get update
$SUDO apt-get install -y git espeak-ng python3-venv
$SUDO apt-get install -y python3.11 python3.11-venv || true
$SUDO apt-get install -y python3.10 python3.10-venv || true

if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
else
  PYTHON_BIN="python3"
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.10|3.11) ;;
  *)
    echo "Unsupported Python version: ${PYTHON_VERSION}" >&2
    echo "Falling back to Miniconda-managed Python 3.11." >&2
    PYTHON_BIN=""
    ;;
esac

if [[ -n "$PYTHON_BIN" ]]; then
  echo "[2/6] Creating Python environment at ${VENV_DIR} with ${PYTHON_BIN}"
  $PYTHON_BIN -m venv "$VENV_DIR"
  source "${VENV_DIR}/bin/activate"
  PYTHON="${VENV_DIR}/bin/python"
  PIP="${PYTHON} -m pip"
  $PIP install --upgrade pip wheel setuptools
else
  echo "[2/6] Installing Miniconda at ${CONDA_DIR}"
  if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    INSTALLER="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
  fi
  CONDA_BIN="${CONDA_DIR}/bin/conda"
  if [[ ! -x "${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
    "$CONDA_BIN" create -y -n "$CONDA_ENV_NAME" python=3.11 pip
  fi
  PYTHON="${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin/python"
  PIP="${PYTHON} -m pip"
  $PIP install --upgrade pip wheel setuptools
fi

echo "[3/6] Installing Python packages"
if ! $PYTHON -c "import torch" >/dev/null 2>&1; then
  $PIP install torch torchvision torchaudio
fi
$PIP install \
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
$PYTHON -c "import yaml, torch; print('Python environment ready')"

echo "[4/6] Generating run config at ${GENERATED_CONFIG}"
$PYTHON - "$BASE_CONFIG" "$GENERATED_CONFIG" "$DATA_DIR" "$LOG_DIR" "$BATCH_SIZE" "$NUM_STEPS" "$SAVE_INTERVAL" "$LOG_INTERVAL" "$MIXED_PRECISION" "$MAX_MEL_LENGTH" "$GRADIENT_CHECKPOINTING" <<'PY'
import sys
import yaml

base_config, output_config, data_dir, log_dir, batch_size, num_steps, save_interval, log_interval, mixed_precision, max_mel_length, gradient_checkpointing = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["data_folder"] = data_dir
config["log_dir"] = log_dir
config["batch_size"] = int(batch_size)
config["num_steps"] = int(num_steps)
config["save_interval"] = int(save_interval)
config["log_interval"] = int(log_interval)
config["mixed_precision"] = mixed_precision
config["gradient_checkpointing"] = gradient_checkpointing.lower() == "true"
config["dataset_params"]["max_mel_length"] = int(max_mel_length)

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "Detected ${GPU_COUNT} GPU(s)"
echo "Using global batch size ${BATCH_SIZE} (${PER_DEVICE_BATCH_SIZE} per device target)"

echo "[5/6] Preprocessing ${LANG_CODE} dataset"
$PYTHON preprocess_ml.py \
  --lang "$LANG_CODE" \
  --config_path "$GENERATED_CONFIG" \
  --root_directory "$SHARD_DIR" \
  --num_shards "$NUM_SHARDS" \
  --n_workers "$N_WORKERS"

echo "[6/6] Starting training"
if [[ "$GPU_COUNT" -gt 1 ]]; then
  "$PYTHON" -m accelerate.commands.launch \
    --num_processes "$GPU_COUNT" \
    --num_machines 1 \
    --mixed_precision "$MIXED_PRECISION" \
    train.py --config_path "$GENERATED_CONFIG"
else
  "$PYTHON" train.py --config_path "$GENERATED_CONFIG"
fi
