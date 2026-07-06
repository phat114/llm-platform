#!/usr/bin/env bash
# Tự dò VRAM của GPU NVIDIA và chọn model phù hợp, ghi vào .env.
# Bật/tắt bằng MODEL_AUTODETECT trong .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
EXAMPLE="$ROOT/.env.example"

# Tạo .env từ mẫu nếu chưa có
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$EXAMPLE" "$ENV_FILE"
  echo "→ Đã tạo .env từ .env.example (nhớ đổi các *_KEY)."
fi

# Nạp .env hiện tại
set -a; # shellcheck disable=SC1090
source "$ENV_FILE"; set +a

# Cho phép tắt auto-detect để tự chọn model
if [[ "${MODEL_AUTODETECT:-true}" != "true" ]]; then
  echo "MODEL_AUTODETECT=false → giữ nguyên MODEL=${MODEL:-?}"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! Không thấy nvidia-smi. Cần cài NVIDIA driver. Giữ MODEL mặc định trong .env."
  exit 0
fi

# Lấy VRAM từng GPU (MiB); dùng GPU nhỏ nhất làm chuẩn, đếm số GPU
mapfile -t MEMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
GPU_COUNT=${#MEMS[@]}
MIN_MEM=${MEMS[0]}
for m in "${MEMS[@]}"; do (( m < MIN_MEM )) && MIN_MEM=$m; done
VRAM_GB=$(( MIN_MEM / 1024 ))
TOTAL_GB=$(( VRAM_GB * GPU_COUNT ))

echo "Phát hiện: ${GPU_COUNT} GPU × ${VRAM_GB}GB = ${TOTAL_GB}GB tổng VRAM"

# Chọn model theo tổng VRAM (ưu tiên dòng Coder cho tự động hóa build + tool-calling)
if   (( TOTAL_GB >= 64 )); then
  MODEL="Qwen/Qwen2.5-Coder-32B-Instruct";      QUANT="";           MAXLEN=32768
elif (( TOTAL_GB >= 24 )); then
  MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ";  QUANT="awq_marlin"; MAXLEN=16384
elif (( TOTAL_GB >= 16 )); then
  MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ";  QUANT="awq_marlin"; MAXLEN=16384
elif (( TOTAL_GB >= 10 )); then
  # 12GB (vd RTX 3060): 7B AWQ dư VRAM → cho context 32k để đọc file dài
  MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ";   QUANT="awq_marlin"; MAXLEN=32768
else
  echo "!! VRAM ${TOTAL_GB}GB quá thấp cho vLLM. Khuyến nghị GPU >= 16GB."
  exit 1
fi

# Upsert biến vào .env
upsert() {
  local k="$1" v="$2"
  if grep -q "^${k}=" "$ENV_FILE"; then
    sed -i "s|^${k}=.*|${k}=${v}|" "$ENV_FILE"
  else
    echo "${k}=${v}" >> "$ENV_FILE"
  fi
}
upsert MODEL "$MODEL"
upsert SERVED_MODEL_NAME "brain"
upsert QUANTIZATION "$QUANT"
upsert MAX_MODEL_LEN "$MAXLEN"
upsert TENSOR_PARALLEL_SIZE "$GPU_COUNT"
upsert TOOL_CALL_PARSER "hermes"

echo "→ Đã ghi vào .env:"
echo "   MODEL=$MODEL"
echo "   QUANTIZATION=${QUANT:-none}  TP=$GPU_COUNT  MAX_MODEL_LEN=$MAXLEN"
