# Training — Bước 2: QLoRA + multi-LoRA serving + eval

Khung fine-tune **một adapter cho một skill** khi prompt "trượt trần" (xem
[docs/multitask-strategy.md](../docs/multitask-strategy.md)). Triết lý: **giữ 1 base dùng chung,
nhân adapter theo task**; train trên base **bf16**, serve **multi-LoRA** trên vLLM.

> ⚠️ Thư mục này chạy trên **GPU box** (deps train nặng), tách khỏi `agents/` và khỏi serving.
> Dùng venv riêng: `python -m venv .venv-train && pip install -r training/requirements.txt`.

## Pipeline

```
1) prepare_data.py   JSONL nhiều task  ──►  data/mixed/{train,val}.jsonl  (cân bằng + replay)
2) train_qlora.py    train.jsonl + config  ──►  adapters/<skill>/        (QLoRA trên base bf16)
3) [serve]           --lora-modules + litellm model_name  ──►  brain-<skill>
4) eval_matrix.py    eval.jsonl  ──►  ma trận task×metric + CI gate (tool-call ≥95%)
   (tùy chọn) merge_lora.py ──► model liền ──► re-quantize AWQ (đường B)
```

## 1) Chuẩn bị dữ liệu
Định dạng: [data/schema.md](data/schema.md). Trộn + cân bằng (chống negative transfer, thêm replay
chống quên tool-calling):
```bash
cd training
python prepare_data.py --data data/example.jsonl \
    --out-dir data/mixed --temperature 1.7 --max-share 0.35 \
    --min-per-task 300 --val-ratio 0.1 \
    --replay-file data/general.jsonl --replay-frac 0.1   # replay tùy chọn
```
In ra `stats.json`: số mẫu nguồn/mục tiêu mỗi task + tham số. **Kiểm mắt** trước khi train.

## 2) Train adapter (QLoRA, base bf16)
Sửa [configs/extract.yaml](configs/extract.yaml) (r=32, alpha=64, all-linear, 4-bit, mask prompt):
```bash
python train_qlora.py --config configs/extract.yaml
# → adapters/extract/  (VRAM ~5–9GB trên 3060; ~1–3h cho 1–3k mẫu)
```
Mỗi skill = một config + một adapter. Train ≥3 seed (đổi `seed`) để có **noise band** khi so sánh.

## 3) Serve multi-LoRA (1 base + N adapter)
**vLLM** — thêm vào `VLLM_EXTRA_ARGS` trong `.env` (entrypoint đã forward, giữ hermes/prefix caching):
```
--enable-lora --max-loras 3 --max-lora-rank 32 --max-cpu-loras 16 \
--lora-modules extract=/adapters/extract review=/adapters/review
```
Đặt env `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` để hot-add không restart. Mount adapter vào container
(`docker-compose.yml`): `- ./training/adapters:/adapters:ro`.

**LiteLLM** — thêm vào `config/litellm.yaml`, mỗi adapter một `model_name`:
```yaml
  - model_name: brain-extract
    litellm_params:
      model: openai/extract          # trùng tên trong --lora-modules
      api_base: http://vllm:8000/v1
      api_key: os.environ/VLLM_API_KEY
```
**Nối vào agent:** trong `agents/skills.py`, đổi `model="brain"` → `model="brain-extract"` cho skill
tương ứng. Không sửa router/agent.

> ⚠️ **Verify AWQ + LoRA:** adapter train trên base **bf16** nhưng serve trên base **AWQ 4-bit** →
> có thể lệch chất lượng. **Bắt buộc** chạy `eval_matrix.py` so adapter-trên-AWQ với base; nếu tụt
> đáng kể → serve task đó trên base bf16 (server ≥24GB) hoặc dùng đường B (merge → re-quant).

## 4) Eval matrix + CI gate
Eval set JSONL (prompt + kiểu check): `json_valid` | `tool_call_valid` | `contains` | `regex`.
```bash
python eval_matrix.py --eval data/eval.jsonl --model brain-extract --baseline brain
```
In ma trận task×metric kèm cột Δ vs baseline; **cổng cứng** `tool_call_valid ≥ 95%` (exit≠0 nếu rớt)
→ dùng làm **cổng promote** trước khi đổi `skills.py`. Chạy `temperature=0`, batched (nhanh trên vLLM).

## (Tùy chọn) Đường B — merge rồi re-quantize
```bash
python merge_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter ./adapters/extract \
    --out ./models/brain-extract-merged
# rồi re-quantize AWQ (AutoAWQ/llm-compressor), trỏ MODEL trong .env sang bản AWQ,
# giữ SERVED_MODEL_NAME=brain, đặt MODEL_AUTODETECT=false.
```

## Cạm bẫy (nhắc lại từ chiến lược)
- ❌ Train trên `*-AWQ` (script chặn sẵn) — phải là base bf16.
- ❌ Đổi chat template / bỏ `<tool_call>` trong data → hỏng tool-calling. Giữ đúng ChatML Qwen2.5.
- ❌ Bỏ eval decontamination → điểm giả. Loại mẫu train trùng eval.
- ❌ Overfit khi data ít → 1–3 epoch, theo dõi val loss.
- ❌ Nhồi nhiều task xung khắc vào 1 adapter → per-task eval tụt; tách adapter.
