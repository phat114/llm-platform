# Host Validation — Smoke-test toàn hệ thống trên GPU host

Tài liệu này hướng dẫn **kiểm tra thực tế (smoke-test)** toàn bộ LLM Platform **trên server GPU** —
những phần máy dev không chạy được: serve vLLM, agent tool-calling e2e, train QLoRA, serve multi-LoRA,
eval trên GPU.

Kiến trúc nhắc lại: **vLLM** (serve Qwen2.5-Coder-7B-AWQ, tên nội bộ `brain`) → **LiteLLM gateway**
(`:4000`, OpenAI-compatible) → client / **Open WebUI** / **agent** (`agents/`). Fine-tune ở `training/`.

> Quy ước mỗi bước: **(a) lệnh** — chạy gì, **(b) mong đợi** — kết quả đúng trông ra sao,
> **(c) nếu lỗi** — nghĩ tới đâu. Chạy tuần tự; đừng qua bước sau khi bước trước còn đỏ.

Biến dùng chung (thay bằng giá trị thật của bạn):
- `SERVER_IP` — IP của GPU host (nếu chạy tại chỗ dùng `localhost`).
- `LITELLM_MASTER_KEY` — lấy từ `.env` (mặc định placeholder `sk-change-me-please`, **phải đổi**).
- `GATEWAY_URL` = `http://SERVER_IP:4000/v1`.

---

## 0. Tiền đề — platform sống, GPU thấy trong Docker

**(a) Lệnh**
```bash
# GPU phải nhìn thấy trong Docker (nvidia-container-toolkit)
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

cd /path/to/llm-platform
cp .env.example .env          # rồi ĐỔI VLLM_API_KEY và LITELLM_MASTER_KEY
make up                       # tự dò VRAM → chọn model → khởi động
make logs                     # theo dõi tải model (lần đầu tải Qwen khá lâu)
make health                   # test khi model đã sẵn sàng
```

**(b) Mong đợi**
- `nvidia-smi` trong container in ra đúng GPU (vd RTX 3060 12GB).
- `make logs` cho thấy vLLM load xong, dòng kiểu `Application startup complete` / `Uvicorn running`.
- `make health` **xanh** (gateway `:4000` trả lời). Kiểm nhanh danh mục model:
  ```bash
  curl -s http://SERVER_IP:4000/v1/models \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
  ```
  → thấy `brain` (và alias `gpt-4o`, và `brain-pro` nếu đã cấu hình).
- Thử một phát chat qua gateway:
  ```bash
  curl -s http://SERVER_IP:4000/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"brain","messages":[{"role":"user","content":"Xin chào"}]}' | jq '.choices[0].message.content'
  ```
- Chat UI mở được ở `http://SERVER_IP:3000`.

**(c) Nếu lỗi**
- `nvidia-smi` trong container fail → chưa cài/khởi động `nvidia-container-toolkit`, hoặc driver NVIDIA lỗi.
- vLLM không lên / OOM khi load → VRAM không đủ cho model auto-chọn. Xem bảng auto-detect trong
  `README.md`; hạ `GPU_MEMORY_UTILIZATION` (0.90→0.85 nếu GPU cũng chạy desktop), hoặc đặt
  `MODEL_AUTODETECT=false` rồi chỉ định `MODEL` nhỏ hơn.
- `make health` đỏ nhưng vLLM còn đang tải → chờ `make logs` xong rồi thử lại (lần đầu tải model lâu).
- `curl /v1/models` trả 401 → sai `LITELLM_MASTER_KEY` (phải trùng `.env`).

---

## 1. Agent e2e — router chọn skill, tool chạy trong WORKDIR

Agent (`agents/`) dùng **venv riêng**, tách khỏi serving.

**(a) Lệnh**
```bash
cd /path/to/llm-platform/agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Trỏ agent vào gateway + khai báo thư mục được phép thao tác
export LITELLM_MASTER_KEY=sk-...                 # trùng .env của platform
export GATEWAY_URL=http://SERVER_IP:4000/v1
export AGENT_WORKDIR=/path/to/repo-test          # 1 repo test để agent đọc/sửa

# 1.1 In registry (không gọi model)
python agent.py --list-skills

# 1.2 Task chat (fallback mặc định, không tool)
python agent.py "Giải thích ngắn gọn LoRA là gì"

# 1.3 Task build (động tới file → cần tool read/write/shell trong WORKDIR)
python agent.py "Đọc README trong repo, thêm dòng '# smoke-test ok' vào cuối rồi liệt kê thư mục"
```

**(b) Mong đợi**
- `--list-skills` in đúng 4 skill: `build` / `review` / `chat` / `extract`, kèm model route
  (`build`,`chat`,`extract`=`brain`; `review`=`brain-pro`) và tool tương ứng.
- Task 1.2: dòng `Skill = chat (model=brain ...)` — router chọn `chat`, không đụng file.
- Task 1.3: dòng `Skill = build (model=brain ...)`; agent gọi `list_dir`/`read_file`/`write_file`;
  hỏi `y/N` trước mỗi lệnh shell (an toàn). File chỉ thay đổi **bên trong** `AGENT_WORKDIR`.
- In ra `WORKDIR = <AGENT_WORKDIR>` và kết thúc bằng `=== XONG ===` + tóm tắt.

**(c) Nếu lỗi**
- Kết nối/`401` → sai `GATEWAY_URL` hoặc `LITELLM_MASTER_KEY`. Xác nhận Bước 0 còn xanh.
- Router luôn rơi về `chat` cho task build → endpoint không hỗ trợ structured output;
  router fallback `json_schema → json_object → DEFAULT_SKILL(chat)`. Không sao về kỹ thuật,
  nhưng để ép đúng skill dùng `--skill build`. Xác nhận model local trả tool-call chuẩn hermes.
- Agent đòi đọc/ghi ngoài repo → bị `_safe_path` chặn (khóa trong `AGENT_WORKDIR`). Đây là **đúng**.
- Muốn khỏi bị hỏi shell từng lệnh (chỉ khi tin task): thêm `--auto`.

---

## 2. Bật `brain-pro` — route task khó/review sang Claude

`review` route sẵn sang `brain-pro` (xem `skills.py`), cần key ngoài.

**(a) Lệnh**
```bash
# Trên HOST: điền key vào .env của platform
#   ANTHROPIC_API_KEY=sk-ant-...
# (config/litellm.yaml đã có sẵn model_name: brain-pro → anthropic/claude-sonnet-5)

# Nạp lại litellm để đọc key mới
docker compose restart litellm          # hoặc: make restart nếu Makefile có

# Xác nhận gateway thấy brain-pro
curl -s http://SERVER_IP:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'

# Test skill review (ép skill, bỏ qua router) — route sang Claude
cd /path/to/llm-platform/agents && source .venv/bin/activate
python agent.py --skill review "Xem file chính trong repo có lỗi bảo mật/thiết kế gì không"
```

**(b) Mong đợi**
- `/v1/models` liệt kê `brain-pro`.
- Dòng `Skill = review (model=brain-pro ...)`; agent **chỉ đọc** (`read_file`, `list_dir`), không
  sửa file / không chạy lệnh thay đổi hệ thống; báo cáo theo mức ưu tiên (nghiêm trọng → nhỏ).
- Chất lượng nhận xét rõ ràng hơn hẳn `brain` local (đây chính là lý do route sang Claude).

**(c) Nếu lỗi**
- `401`/`authentication` khi gọi `brain-pro` → `ANTHROPIC_API_KEY` trống/sai, hoặc chưa
  `restart litellm` sau khi điền key.
- Muốn model mạnh hơn: đổi `anthropic/claude-sonnet-5` → `claude-opus-4-8` trong `config/litellm.yaml`.
- (Tùy chọn) auto-fallback khi `brain` local OOM/timeout: bỏ comment
  `fallbacks: [{"brain": ["brain-pro"]}]` trong `litellm.yaml` — **chỉ sau khi** đã có key,
  không thì lỗi 401.

---

## 3. Smoke QLoRA (GPU) — train 1 epoch ra adapter

Thư mục `training/` chạy trên **GPU box**, **venv riêng** (deps train nặng).

**(a) Lệnh**
```bash
cd /path/to/llm-platform/training
python -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements.txt

# Tạo dataset nhỏ (vài chục dòng) theo đúng data/schema.md, lưu vd: data/smoke.jsonl
#   → xem data/schema.md cho format (ChatML Qwen2.5, giữ <tool_call> nếu có tool).

# 1) Trộn + cân bằng
python prepare_data.py --data data/smoke.jsonl \
    --out-dir data/mixed --temperature 1.7 --max-share 0.35 \
    --min-per-task 300 --val-ratio 0.1
# → kiểm mắt data/mixed/stats.json + data/mixed/{train,val}.jsonl

# 2) Train QLoRA (base bf16), ép epochs=1 cho smoke-test
python train_qlora.py --config configs/extract.yaml   # sửa epochs=1 trong config trước khi chạy
```

**(b) Mong đợi**
- `prepare_data.py` sinh `data/mixed/{train,val}.jsonl` + `stats.json` (số mẫu nguồn/mục tiêu mỗi task).
- `train_qlora.py` chạy trọn 1 epoch, val loss in ra hợp lý, **sinh thư mục `adapters/extract/`**
  (chứa adapter LoRA + config).
- **VRAM ~5–9GB** trên 3060 trong lúc train (theo dõi bằng `nvidia-smi`).

**(c) Nếu lỗi**
- **Train nhầm base `*-AWQ`**: script **chặn sẵn** — phải train trên **base bf16**
  (`Qwen/Qwen2.5-Coder-7B-Instruct`), không phải bản AWQ đang serve. Sửa `base` trong config.
- **OOM**: giảm batch size / bật gradient accumulation / giảm `max_seq_len` trong config;
  đảm bảo vLLM không đang giữ hết VRAM (có thể `make down` để nhường GPU khi train).
- Val loss không giảm / overfit khi data ít → giữ **1–3 epoch**, theo dõi val loss.
- Hỏng tool-calling về sau → do đổi chat template / bỏ `<tool_call>` trong data. Giữ đúng ChatML Qwen2.5.

---

## 4. Serve multi-LoRA — 1 base + adapter, thấy `brain-extract`

**(a) Lệnh**
```bash
# .env: thêm cờ LoRA vào VLLM_EXTRA_ARGS (giữ nguyên các cờ cũ như --enable-prefix-caching)
#   VLLM_EXTRA_ARGS=--enable-prefix-caching \
#     --enable-lora --max-loras 3 --max-lora-rank 32 --max-cpu-loras 16 \
#     --lora-modules extract=/adapters/extract
#
# docker-compose.yml: mount adapter vào container vllm (nếu chưa có)
#   volumes:
#     - ./training/adapters:/adapters:ro

# config/litellm.yaml: BỎ COMMENT khối brain-extract
#   - model_name: brain-extract
#     litellm_params:
#       model: openai/extract          # PHẢI trùng tên trái trong --lora-modules (extract=...)
#       api_base: http://vllm:8000/v1
#       api_key: os.environ/VLLM_API_KEY

# Khởi động lại serving
make down && make up          # (hoặc docker compose up -d vllm litellm)

# Kiểm model xuất hiện
curl -s http://SERVER_IP:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
```

**(b) Mong đợi**
- `/v1/models` liệt kê **`brain-extract`** (bên cạnh `brain`).
- 3 chuỗi phải khớp nhau: `--lora-modules extract=/adapters/extract` ↔ `model: openai/extract`
  ↔ `model_name: brain-extract`.
- Gọi thử adapter:
  ```bash
  curl -s http://SERVER_IP:4000/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"brain-extract","messages":[{"role":"user","content":"Trích JSON từ: ..."}]}' \
    | jq '.choices[0].message.content'
  ```

**(c) Nếu lỗi**
- `brain-extract` không xuất hiện → chưa bỏ comment trong `litellm.yaml`, hoặc chưa restart litellm.
- vLLM log báo không tìm thấy adapter → sai đường mount `/adapters/extract` (kiểm volume trong
  `docker-compose.yml`) hoặc tên trong `--lora-modules` không khớp thư mục.
- Model gọi được nhưng lỗi rank → `--max-lora-rank` phải ≥ `r` của adapter (config train dùng r=32).
- (Tùy chọn) hot-add adapter không restart: đặt env `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`.

---

## 5. Verify AWQ + LoRA + eval — CI gate tool_call ≥ 95%

⚠️ Adapter train trên base **bf16** nhưng đang serve trên base **AWQ 4-bit** → **có thể lệch chất lượng**.
Bước này **bắt buộc**: so `brain-extract` với baseline `brain`.

**(a) Lệnh**
```bash
cd /path/to/llm-platform/training && source .venv-train/bin/activate

# Eval set JSONL: prompt + kiểu check (json_valid | tool_call_valid | contains | regex)
python eval_matrix.py --eval data/eval.jsonl --model brain-extract --baseline brain
echo "exit=$?"
```

**(b) Mong đợi**
- In **ma trận task × metric** kèm cột **Δ vs baseline**.
- **Cổng cứng** `tool_call_valid ≥ 95%` → **exit code = 0** khi đạt (dùng làm **cổng promote**
  trước khi đổi `model=` trong `agents/skills.py`).
- Chạy ở `temperature=0`, batched (nhanh trên vLLM).

**(c) Nếu lỗi / quyết định**
- **exit ≠ 0** (rớt gate tool_call) hoặc adapter-trên-AWQ **tụt đáng kể** so baseline → cân nhắc:
  - **Đường A**: serve **task đó** trên **base bf16** (yêu cầu server ≥ 24GB VRAM), giữ AWQ cho phần còn lại.
  - **Đường B**: `merge_lora.py` (merge adapter vào base) → **re-quantize AWQ** → trỏ `MODEL` sang bản
    merged, giữ `SERVED_MODEL_NAME=brain`, đặt `MODEL_AUTODETECT=false`.
- Điểm cao giả tạo → thiếu **decontamination**: loại mẫu train trùng eval.
- Chỉ promote (đổi `skills.py` `model="brain"` → `"brain-extract"`) **sau khi** gate xanh.

---

## Tổng kết luồng

```
0 make up + health (GPU trong docker)  →  1 agent e2e (chat + build, WORKDIR)
  →  2 brain-pro (review → Claude)  →  3 QLoRA smoke (adapters/extract, VRAM 5–9GB)
  →  4 serve multi-LoRA (brain-extract trong /v1/models)  →  5 eval + CI gate → promote
```

Xanh hết 0→5 nghĩa là chuỗi serve → agent → train → multi-LoRA → eval đã thông trên GPU host.
