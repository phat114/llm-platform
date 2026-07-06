#!/usr/bin/env bash
# Lắp ráp tham số vLLM từ biến môi trường (.env). Xử lý quantization tùy chọn.
set -euo pipefail

ARGS=(
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME:-brain}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --max-model-len "${MAX_MODEL_LEN:-16384}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --host 0.0.0.0
  --port 8000
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER:-hermes}"
)

# Chỉ thêm --quantization khi có (model full-precision thì để trống)
[[ -n "${QUANTIZATION:-}" ]] && ARGS+=(--quantization "${QUANTIZATION}")
# Key bảo vệ vLLM (gateway sẽ dùng key này)
[[ -n "${VLLM_API_KEY:-}" ]] && ARGS+=(--api-key "${VLLM_API_KEY}")
# Cờ bổ sung tùy ý
# shellcheck disable=SC2206
[[ -n "${VLLM_EXTRA_ARGS:-}" ]] && ARGS+=(${VLLM_EXTRA_ARGS})

echo "Khởi động vLLM với: ${ARGS[*]}"
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
