# Kiến trúc — llm-platform

> Đọc file này trước khi sửa `docker-compose.yml`, `config/litellm.yaml` hay `.env`.
> Nó trả lời: *ai làm gì*, *key nào dùng ở đâu*, và *những cái bẫy đã cắn thật* khi chạy trên máy dev.

## 1. Điều dễ hiểu nhầm nhất: repo này KHÔNG có API của riêng nó

Không có FastAPI, không có route, không có `main.py`. Grep `@app.` / `APIRouter` ra **0 kết quả**.

Mọi endpoint bạn thấy trong docs (`/v1/chat/completions`, `/v1/models`, `/v1/load_lora_adapter`…)
đều **do image dựng sẵn cung cấp**. Bạn không *viết* API — bạn **cấu hình** chúng.

Vì vậy `docker compose build` gần như là no-op: không service nào có `build:`, cả ba đều `image:` kéo về.
Lệnh đúng là `docker compose pull` rồi `up -d` (hoặc `make up`).

Toàn bộ "code" của repo chia làm hai nhóm:

| Nhóm | File | Vai trò |
|---|---|---|
| **Hạ tầng** (dán 3 image lại) | `docker-compose.yml`, `config/litellm.yaml`, `scripts/*.sh` | Khai báo, không thực thi logic |
| **Client** (gọi API, không phục vụ API) | `agents/*.py`, `training/*.py` | Dùng OpenAI SDK gọi vào `localhost:4000` như mọi app bên ngoài |

Nếu sau này cần API riêng (`/api/summarize`, có DB, có business logic) → đó là **service mới**, và nó
sẽ gọi vào `:4000` giống hệt mọi client khác. Đừng nhét vào repo này.

## 2. Ba tầng — mỗi tầng một việc

```
Client / App / Agent build
        │  (OpenAI API)  Bearer = LITELLM_MASTER_KEY
   ┌────▼─────┐   ghcr.io/berriai/litellm     :4000
   │ litellm  │   auth · log · routing · dịch protocol OpenAI↔Anthropic
   └────┬─────┘
        │  Bearer = VLLM_API_KEY
   ┌────▼─────┐   vllm/vllm-openai            :8000
   │  vllm    │   nạp weights lên GPU · sinh token · PagedAttention · batching
   └──────────┘

   Open WebUI  ghcr.io/open-webui/open-webui  :3000  → gọi litellm như một client
```

- **vLLM** — động cơ thật. Cái đáng giá: PagedAttention (KV cache phân trang, không phí VRAM),
  continuous batching (5 request gộp chung 1 batch), prefix caching. Tự viết = nhiều tháng.
- **LiteLLM** — gateway. Giá trị lớn nhất **không phải** proxy, mà là **dịch protocol**: `brain-pro`
  là Claude (Anthropic Messages API — protocol khác hẳn), nhưng client vẫn chỉ dùng một OpenAI SDK,
  đổi `model=` là xong. Ngoài ra: `master_key`, alias (`gpt-4o` → model local, cho app hardcode),
  `drop_params`, `num_retries`, `fallbacks`.
- **Open WebUI** — clone ChatGPT chạy local. Có DB riêng (volume `open-webui`) → **nhớ model của
  từng cuộc chat**. Đây là lý do đổi model ở dropdown mà chat cũ vẫn lỗi: phải mở chat MỚI.

Mỗi tầng thay được độc lập: bỏ WebUI thì API vẫn chạy; bỏ LiteLLM thì gọi thẳng vLLM `:8000` vẫn ra token.

**Cạm bẫy về `fallbacks`:** nó chỉ **phản ứng khi lỗi** (OOM/timeout), KHÔNG biết task nào khó.
Muốn "task khó → Claude" thì phải quyết định ở tầng ứng dụng (`agents/skills.py`). Xem
[multitask-strategy.md](multitask-strategy.md).

## 3. Hai key — dùng ở đâu, khi nào

| Key | Chặng nó bảo vệ | Bạn có phải gõ tay không |
|---|---|---|
| `LITELLM_MASTER_KEY` | client → LiteLLM `:4000` | **Có.** Đây là key duy nhất bạn dùng hằng ngày |
| `VLLM_API_KEY` | LiteLLM → vLLM `:8000` | **Không.** Hai container tự khớp qua `env_file: .env` |

`LITELLM_MASTER_KEY` được đọc ở: `config/litellm.yaml` (`master_key`), `docker-compose.yml` (tiêm vào
Open WebUI làm `OPENAI_API_KEY`), `scripts/healthcheck.sh`, `agents/agent.py`, `training/distill_data.py`,
`training/eval_matrix.py`, `Jenkinsfile`.

⚠️ Đám script Python/bash đọc key từ **biến môi trường của shell**, KHÔNG tự đọc `.env`. Chạy ngoài
Docker thì phải `export LITELLM_MASTER_KEY=...` trước.

**Vì sao chặng nội bộ vẫn cần key?** Vì `:8000` có publish ra ngoài. vLLM trần không có log, không
quota, không phân biệt người dùng — ai gọi thẳng được `:8000` là **bypass toàn bộ gateway**. Chỉ có
một trường hợp gõ `VLLM_API_KEY` bằng tay: gọi API riêng của vLLM mà gateway không proxy, ví dụ
`/v1/load_lora_adapter` (xem [fine-tuning-playbook.md](fine-tuning-playbook.md)).

## 4. Mạng — ai được lộ ra LAN

Điều khiển bằng 3 biến trong `.env`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `BIND_ADDR` | `0.0.0.0` | Địa chỉ bind cho **service public** (Chat UI + Gateway). Đặt `127.0.0.1` để rút cả stack về localhost |
| `VLLM_BIND` | `127.0.0.1` | vLLM là **engine thô** → giữ localhost. Client LAN phải đi qua gateway `:4000` để còn có master-key + retry + fallback |
| `LAN_HOST` | `thinhphat.local` | Hostname mDNS cố định (IP là DHCP, đổi bất chợt). Open WebUI cần nó để sinh link tuyệt đối |

Nguyên tắc: **cổng public duy nhất là gateway.** Engine thô không bao giờ lộ thẳng ra ngoài.

## 5. Chọn model — auto-detect và khi nào tắt nó

`scripts/detect-gpu.sh` dò VRAM qua `nvidia-smi` rồi **`sed` ghi đè thẳng vào `.env`**
(`MODEL`, `QUANTIZATION`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`). Nó chạy trên **host**, không
liên quan gì tới Docker — `docker compose up` KHÔNG gọi nó. Chỉ hai đường:

- `make detect` → chỉ dò, ghi `.env`, không khởi động gì
- `make up` → `start.sh` gọi detect **rồi mới** `docker compose up -d`

Ngưỡng chọn theo **tổng VRAM** = (VRAM GPU nhỏ nhất) × (số GPU):

| Tổng VRAM | Model | Quant |
|---|---|---|
| ≥ 64 GB | Qwen2.5-Coder-32B-Instruct | bf16 |
| ≥ 24 GB | Qwen2.5-Coder-32B-Instruct-AWQ | awq_marlin |
| ≥ 16 GB | Qwen2.5-Coder-14B-Instruct-AWQ | awq_marlin |
| ≥ 10 GB | Qwen2.5-Coder-7B-Instruct-AWQ | awq_marlin |
| **< 10 GB** | **`exit 1`** — script không có nhánh nào | — |

⚠️ `start.sh` **nuốt lỗi** của detect (`|| echo`). Nên trên máy < 10GB, `make up` vẫn chạy tiếp với
`MODEL` cũ trong `.env` → OOM. Đặt `MODEL_AUTODETECT=false` rồi chỉnh tay là đường đúng.

## 6. Cạm bẫy đã cắn thật (máy dev: GTX 1660 SUPER 6GB, Windows + WSL2)

Bốn thứ dưới đây đều đã làm container `vllm` chết hoặc chat báo lỗi. Cấu hình hiện tại trong `.env`
đã né hết — **đừng gỡ mà không hiểu vì sao**.

**a) `make` không chạy trên PowerShell.** Cả stack là bash + Docker. Phải vào WSL:
`wsl -d Ubuntu` → `cd /mnt/g/work/llm-platform` → `make up`.

**b) `RuntimeError: UVA is not available` — chết ngay lúc `init_device`.**
vLLM ≥ 0.25 mặc định dùng **V2 model runner**, nó cấp phát `UvaBuffer` → cần Unified Virtual
Addressing. GPU trong WSL2 đi qua paravirtualization `dxgkrnl`, **không có UVA**.
→ `VLLM_USE_V2_MODEL_RUNNER=0` trong `.env` (ép về V1 runner, không đụng UVA).
**Trên server Ubuntu thật thì BỎ dòng này** — V2 nhanh hơn.

**c) Turing (SM 7.5) không phải Ampere.** GTX 16xx/20xx dính cả ba:
- `awq_marlin` cần SM ≥ 8.0 → dùng `QUANTIZATION=awq` thường.
- **Không có bfloat16** → bắt buộc `--dtype half` trong `VLLM_EXTRA_ARGS`, thiếu là chết lúc start.
- FlashAttention-2 / FlashInfer cần SM ≥ 8.0 → vLLM tự fallback `TRITON_ATTN`. Log in ra dòng
  `ERROR ... FA2 is only supported on devices with compute capability >= 8` — **đây là cảnh báo, không phải lỗi.**
  Hệ quả: chậm, ~8–10 token/s.

**d) 6GB VRAM chỉ đủ cho MỘT engine.** vLLM preallocate KV cache (`GPU_MEMORY_UTILIZATION=0.80`
→ giữ chặt ~4.8GB). Ollama vào sau sẽ báo
`model requires more system memory (4.9 GiB) than is available (3.0 GiB)`.
→ Đã đặt `ENABLE_OLLAMA_API: "false"` cho Open WebUI để model Ollama không hiện trong dropdown.
Muốn dùng Ollama thì phải `make down` vLLM trước. **Chọn một, không chạy song song.**

Cấu hình `.env` đang chạy được trên máy này: `Qwen2.5-Coder-3B-Instruct-AWQ` + `awq` + `--dtype half`
+ `MAX_MODEL_LEN=8192` + `GPU_MEMORY_UTILIZATION=0.80` + `VLLM_USE_V2_MODEL_RUNNER=0`.
Nói thẳng: **3B đủ để test luồng, không đủ để làm "bộ não" thật.** Việc khó → `brain-pro`.

## 7. Tra API ở đâu

**Swagger sống trên máy — chính xác 100% với version đang chạy, tốt hơn mọi link web:**

| | URL | Ghi chú |
|---|---|---|
| vLLM | http://localhost:8000/docs | API engine |
| LiteLLM | http://localhost:4000/ | Swagger ở **trang gốc**, không phải `/docs`. ~538 route: `/key/generate`, `/spend/*`, `/model/info`, `/health/*` |

Đám `/key/*` và `/spend/*` chỉ lưu được khi bật `database_url` trong `config/litellm.yaml`.

**Docs chính thức:**
- LiteLLM config: https://docs.litellm.ai/docs/proxy/configs · virtual keys:
  https://docs.litellm.ai/docs/proxy/virtual_keys · routing/fallback:
  https://docs.litellm.ai/docs/routing-load-balancing
- vLLM: https://docs.vllm.ai — cần nhất là *engine args*, *env vars*, *tool calling*, *LoRA*
- Open WebUI env: https://docs.openwebui.com/getting-started/env-configuration
- Protocol mà cả hai đều nói (OpenAI Chat Completions): https://platform.openai.com/docs/api-reference/chat

## 8. Bản đồ repo

| Đường dẫn | Là gì |
|---|---|
| `docker-compose.yml` | Lắp 3 image. Không build gì |
| `config/litellm.yaml` | Khai báo model: `brain` (local), `gpt-4o` (alias), `brain-pro` (Claude). Thêm model = thêm ~5 dòng YAML |
| `scripts/detect-gpu.sh` | Dò VRAM → `sed` ghi `.env`. Chạy trên host |
| `scripts/vllm-entrypoint.sh` | Ráp tham số CLI cho vLLM từ `.env` (mount vào container) |
| `scripts/healthcheck.sh` | `make health` — test 3 tầng |
| `agents/` | **Client**: vòng lặp tool-calling, router, RAG |
| `training/` | **Client**: distill data, QLoRA, eval, merge |
| `docs/` | [fine-tuning-playbook.md](fine-tuning-playbook.md) · [multitask-strategy.md](multitask-strategy.md) · [host-validation.md](host-validation.md) |
