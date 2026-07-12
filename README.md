# LLM Platform

Bộ não LLM tự host bằng **vLLM**, expose một endpoint **OpenAI-compatible** duy nhất cho: chat, agent tự động hóa build, và các app khác.

```
Client / App / Agent build
        │  (OpenAI API)
   ┌────▼─────┐   LiteLLM gateway  — auth, key, log, routing
   │ litellm  │───┐
   └────┬─────┘   │  ┌──────────┐   Postgres — virtual key + spend + admin UI (:4000/ui)
        │         └──│ postgres │
        │            └──────────┘
   ┌────▼─────┐   vLLM             — chạy model trên GPU, tool-calling
   │  vllm    │
   └──────────┘
   Open WebUI  — giao diện chat
```

> **Chi tiết kiến trúc + cạm bẫy khi chạy: [docs/architecture.md](docs/architecture.md).**
> Đọc trước khi sửa `docker-compose.yml` / `config/litellm.yaml` / `.env`, hoặc khi `vllm` báo Error.

## Yêu cầu (trên server GPU)
- Ubuntu + GPU NVIDIA
- NVIDIA driver + `nvidia-container-toolkit`
- Docker + Docker Compose plugin

Kiểm tra GPU thấy trong Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Chạy
```bash
cp .env.example .env      # rồi ĐỔI VLLM_API_KEY và LITELLM_MASTER_KEY
make up                   # tự dò VRAM → chọn model → khởi động
make logs                 # xem tiến trình tải model (lần đầu lâu)
make health               # test khi đã sẵn sàng
```
- Chat UI: http://SERVER_IP:3000
- API gateway: http://SERVER_IP:4000/v1  (Bearer = `LITELLM_MASTER_KEY`)

## Truy cập từ LAN
Chat UI (`:3000`) và gateway (`:4000`) bind `0.0.0.0` → máy khác trong LAN vào được.
vLLM (`:8000`) giữ localhost-only — LAN đi qua gateway. Đặt `LAN_HOST=<hostname>.local`
trong `.env` để dùng **hostname cố định** thay IP DHCP hay đổi.

Windows còn phải mở firewall (chạy bằng quyền Administrator):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1
```
Chi tiết + xử lý sự cố: [docs/lan-access.md](docs/lan-access.md).

## Auto chọn model
`scripts/detect-gpu.sh` dò VRAM và ghi `MODEL`, `QUANTIZATION`, `TENSOR_PARALLEL_SIZE`... vào `.env`:

| Tổng VRAM | Model tự chọn |
|-----------|---------------|
| ≥ 64 GB   | Qwen2.5-Coder-32B-Instruct (bf16) |
| ≥ 24 GB   | Qwen2.5-Coder-32B-Instruct-AWQ |
| ≥ 16 GB   | Qwen2.5-Coder-14B-Instruct-AWQ |
| ≥ 10 GB   | Qwen2.5-Coder-7B-Instruct-AWQ |

Muốn tự chọn model: đặt `MODEL_AUTODETECT=false` trong `.env` rồi sửa `MODEL`.

## Gọi từ code (OpenAI SDK)
```python
from openai import OpenAI
client = OpenAI(base_url="http://SERVER_IP:4000/v1", api_key="<LITELLM_MASTER_KEY>")
r = client.chat.completions.create(
    model="brain",
    messages=[{"role": "user", "content": "Viết hàm Python đảo chuỗi"}],
)
print(r.choices[0].message.content)
```

## Model
- `brain` — local trên GPU (RTX 3060 → Qwen2.5-Coder-7B). Nhanh, miễn phí, việc nhẹ.
- `brain-pro` — model mạnh qua API ngoài (điền `ANTHROPIC_API_KEY`) cho task khó. Xem `config/litellm.yaml`.

## Lộ trình
- [x] Phase 1 — vLLM serving (tool-calling)
- [x] Phase 2 — Gateway (LiteLLM: key, log, multi-model, hybrid local+API)
- [x] Phase 3 — Chat UI (Open WebUI)
- [x] Phase 4 — Agent tự động hóa build (`agents/` — vòng lặp tool-calling)
 cd G:\work\llm-platform
>> powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1

cd G:\work\llm-platform
powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1