#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
docker compose down
echo "✔ Đã dừng. (Model cache vẫn giữ trong volume hf-cache, lần sau khởi động nhanh.)"
