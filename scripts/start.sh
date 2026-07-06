#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Bước 1: Auto-detect GPU và chọn model"
bash scripts/detect-gpu.sh || echo "(bỏ qua auto-detect, dùng MODEL trong .env)"

echo ""
echo "==> Bước 2: Khởi động services"
docker compose up -d

# Nạp cổng để in link
set -a; source .env; set +a
echo ""
echo "✔ Đang tải model & khởi động (LẦN ĐẦU có thể mất nhiều phút do tải weights)."
echo "  Chat UI  : http://localhost:${WEBUI_PORT:-3000}"
echo "  Gateway  : http://localhost:${LITELLM_PORT:-4000}  (dùng LITELLM_MASTER_KEY)"
echo "  vLLM API : http://localhost:${VLLM_PORT:-8000}/v1"
echo ""
echo "  Xem log tải model : docker compose logs -f vllm"
echo "  Kiểm tra sức khỏe : make health"
