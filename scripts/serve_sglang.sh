#!/usr/bin/env bash
# scripts/serve_sglang.sh
# Usage: bash scripts/serve_sglang.sh [model_repo] [adapter_path] [port]
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

MODEL_REPO="${1:-Qwen/Qwen2.5-1.5B}"
ADAPTER_PATH="${2:-}"
PORT="${3:-30000}"

N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TP_SIZE=1
if [ "${N_GPU}" -gt 1 ]; then
    TP_SIZE=${N_GPU}
    echo "Detected ${N_GPU} GPUs — enabling tensor parallelism (tp-size=${TP_SIZE})"
else
    echo "Single GPU detected — running without tensor parallelism"
fi

CMD=(python -m sglang.launch_server --model-path "${MODEL_REPO}" --port "${PORT}" --tp-size "${TP_SIZE}")

if [ -n "${ADAPTER_PATH}" ]; then
    CMD+=(--lora-paths "default=${ADAPTER_PATH}")
    echo "Serving with LoRA adapter: ${ADAPTER_PATH}"
fi

echo "Launching: ${CMD[*]}"
exec "${CMD[@]}"
