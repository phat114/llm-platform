#!/usr/bin/env bash
# Kiểm tra nhanh cả 3 tầng.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; source .env; set +a

echo "== [1] vLLM /health =="
curl -sf "http://localhost:${VLLM_PORT:-8000}/health" && echo "  OK" || echo "  chưa sẵn sàng (còn đang tải model?)"

echo ""
echo "== [2] Gateway /v1/models =="
curl -s "http://localhost:${LITELLM_PORT:-4000}/v1/models" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" | head -c 600
echo ""

echo ""
echo "== [3] Test chat + tool-calling =="
curl -s "http://localhost:${LITELLM_PORT:-4000}/v1/chat/completions" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "brain",
    "messages": [{"role":"user","content":"Trả lời ngắn gọn: bạn là model gì?"}],
    "max_tokens": 128
  }' | head -c 1200
echo ""
