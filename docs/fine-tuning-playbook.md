# Cẩm nang Tinh chỉnh `brain` (Qwen2.5-Coder-7B-AWQ) trên nền vLLM + LiteLLM

> Tài liệu hợp nhất cho stack cụ thể của bạn: `brain` = **Qwen/Qwen2.5-Coder-7B-Instruct-AWQ** (7B, AWQ 4-bit) serve bằng **vLLM** (OpenAI-compatible, `--tool-call-parser hermes`, `--enable-auto-tool-choice`, `MAX_MODEL_LEN=32768`, `GPU_MEMORY_UTILIZATION=0.90`, `TENSOR_PARALLEL_SIZE=1`, prefix caching bật) → gateway **LiteLLM** expose `brain` (local GPU) + `brain-pro` (Claude API). Dev/deploy hiện tại: **RTX 3060 12GB**; server production có thể **24–48GB** (script `detect-gpu.sh` chọn model lớn hơn khi ≥24GB / ≥64GB). Use case: trợ lý code + agent tool-calling (`read_file/write_file/list_dir/run_shell`), **output tiếng Việt**. Triết lý: self-host, gọn nhẹ, ít phụ thuộc framework nặng.

---

## 1. Mở đầu: "tinh chỉnh AI" là một phổ, không phải một nút bấm

Khi người ta nói "fine-tune model", họ thường tưởng tượng phải train lại trọng số. Thực tế, việc điều chỉnh hành vi một LLM trải trên **một phổ liên tục** từ **không đụng một gram trọng số** cho tới **train nặng nhiều GPU**:

```
KHÔNG train  ──────────────────────────────────►  Train nặng
Prompt/Decoding → RAG → PEFT (LoRA/QLoRA) → Preference (DPO/ORPO) → Full-SFT / CPT
   rẻ, phút          giờ           giờ→ngày             ngày → tuần, cụm GPU
```

Nguyên tắc vàng: **chọn bậc thấp nhất giải quyết được nhu cầu**, không phải bậc "oách" nhất. Với một model **7B đã quant 4-bit** chạy trên **12GB VRAM**, mỗi điểm phần trăm chất lượng đều quý — và phần lớn nhu cầu thực tế (định dạng output, giọng tiếng Việt, độ tin cậy JSON, nạp tri thức codebase) **giải quyết được mà không cần train**. Fine-tune chỉ nên đụng tới khi các tầng rẻ hơn đã cạn.

Phân biệt cốt lõi để chọn đúng tầng:

| Bạn cần thêm gì cho model? | Tầng đúng |
|---|---|
| **Định dạng / kỷ luật thao tác / khuôn output** | Prompt + decoding + guided decoding |
| **Tri thức/sự kiện** (codebase, API nội bộ, docs thay đổi liên tục) | **RAG** (không train) |
| **Hành vi/văn phong/format bền vững** (giọng tiếng Việt, format tool-call ổn định) | **PEFT** (LoRA/QLoRA) |
| **"Khẩu vị" chọn câu trả lời tốt hơn** | **Preference optimization** (DPO/ORPO/KTO) |
| **Năng lực nền hoàn toàn mới / ngôn ngữ mới** | Full-SFT / CPT (hầu như **loại trừ** trên phần cứng của bạn) |

---

## 2. Cây quyết định cho ĐÚNG setup của bạn

Với Qwen-Coder-7B-AWQ + RTX 3060 12GB + agent code tiếng Việt, thứ tự ưu tiên khuyến nghị:

```
1. Prompt / Context / Decoding  (guided decoding là cú hích lớn nhất cho độ tin cậy agent)
        │  còn thiếu TRI THỨC codebase?
        ▼
2. RAG  (thêm embedding + vector store; KHÔNG train brain)
        │  còn thiếu HÀNH VI/ĐỊNH DẠNG bền vững?
        ▼
3. QLoRA-SFT  (đường ray train duy nhất vừa 12GB; ~5–9GB VRAM)
        │  cần căn chỉnh "khẩu vị" sau SFT?
        ▼
4. QLoRA-DPO / ORPO  (ORPO gộp luôn SFT trong 1 bước)

   ✗ Full fine-tune & Continued Pre-training: gần như LOẠI TRỪ trên 3060 và cả server 24–48GB đơn GPU.
```

**Vì sao thứ tự này:**
- Tầng 1 và 2 **không train**, ROI cao nhất, reversible tức thì, chạy ngay trên 12GB.
- Tầng 3 dùng **QLoRA** vì LoRA 16-bit thuần cần base bf16 (~15GB) — **không vừa 12GB**; QLoRA nén base xuống 4-bit để train vừa GPU nhỏ.
- Tầng 4 chỉ tinh chỉnh khẩu vị, **không dạy kiến thức mới** — làm sau SFT (hoặc dùng ORPO để gộp).
- Full-FT/CPT cần **~16x số bytes trên mỗi tham số** (xem mục 3.4) → hàng trăm GB, bất khả thi trên phần cứng của bạn.

Xuyên suốt: giữ kiến trúc **hybrid** — câu khó multi-step vẫn route sang `brain-pro` (Claude). Fine-tune chỉ nâng `brain` local cho việc nhẹ/định dạng/giọng văn, không kỳ vọng thay model lớn.

---

## 3. Chi tiết từng nhóm phương pháp

### 3.A — Không train: Prompt & Context Engineering & Decoding

Tầng **weights-frozen**: giữ nguyên checkpoint AWQ 4-bit, chỉ can thiệp **input** (prompt/context) và **logits lúc decode**. ROI cao nhất, phải khai thác cạn kiệt TRƯỚC khi nghĩ tới fine-tune.

**Chi phí chung cho cả nhóm** (không phương pháp nào train → phần "VRAM/thời gian train" đều N/A):

| Hạng mục | RTX 3060 12GB | Server 24–48GB |
|---|---|---|
| Weights `brain` (7B AWQ 4-bit) | ~5–6 GB | ~5–6 GB (hoặc 14B/32B-AWQ theo `detect-gpu.sh`) |
| KV cache (28 layers, 4 KV heads GQA, head_dim 128, fp16) | ~56 KB/token → **32k ≈ 1.8 GB/sequence** | như nhau |
| VRAM tăng thêm khi dùng nhóm này | **~0** (chỉ tốn thêm token prompt = ăn KV cache) | ~0 |
| Chi phí train | **Không có** | **Không có** |

Khác biệt duy nhất giữa 3060 và server là **throughput/concurrency** và **cỡ model nền**, không phải khả năng dùng kỹ thuật.

#### 1) System Prompt Design
Định hình vai trò, ràng buộc, format, ngôn ngữ (output tiếng Việt), kỷ luật thao tác agent (khảo sát trước khi sửa, tự chạy test). Bản hiện tại ở biến `SYSTEM` trong `agents/agent.py`.
- **Không train.** Chỉ cần 1 văn bản 200–1500 tokens viết tay. Với 7B nên viết **ngắn, dứt khoát, đánh số bước** (model nhỏ dễ lạc trong prompt dài).
- Áp mặc định cho mọi client: đặt trong `config/litellm.yaml`. Sửa `SYSTEM` trong `agent.py` **không cần restart vLLM**.
- **Công sức: THẤP.** Ưu: rẻ nhất, reversible, đòn bẩy lớn nhất cho model nhỏ. Nhược: không dạy được kiến thức model chưa có.

#### 2) Few-shot / In-Context Learning
Nhét 3–5 cặp `input → output` mẫu để model bắt chước **format/style**. Rất mạnh để ép định dạng output tiếng Việt nhất quán.
- Mỗi ví dụ ~200–800 tokens; 5 ví dụ ≈ 1–4k tokens. Với `MAX_MODEL_LEN=32768`, 5 ví dụ + system (~5k) vẫn còn ~27k cho nội dung file.
- **Đặt CỐ ĐỊNH ngay sau system prompt** (thứ tự: system → few-shot → nội dung động) để **prefix caching** tái dùng KV.
- **Công sức: THẤP → VỪA.**

#### 3) Output Steering
Ép nhẹ đầu ra không qua grammar cứng: **prefill** (mồi đầu câu trả lời), **logit_bias** (map tokenId→bias −100..100), **stop sequences**.
- Tra token id bằng `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")`.
- `stop`, `logit_bias` là param OpenAI chuẩn → LiteLLM forward. Với agent tool-calling hermes: **không prefill đè** lên cơ chế tool-call.
- **Công sức: THẤP → VỪA.**

#### 4) Guided / Constrained Decoding — **QUAN TRỌNG NHẤT cho agent**
Ép output tuân thủ **cấu trúc cứng**: JSON schema, regex, grammar EBNF. Cơ chế: mỗi bước decode, backend mask logits các token vi phạm về `-inf` → model **không thể** sinh sai cú pháp. Đây là cách nâng độ tin cậy 7B lên gần model lớn **không đổi phần cứng, không fine-tune**. Tương thích hoàn toàn AWQ.

| Cách khai báo | Param vLLM | Chuẩn OpenAI | Dùng khi |
|---|---|---|---|
| JSON theo schema | `guided_json` | `response_format={"type":"json_schema",...}` | Field cố định |
| Chọn 1 trong tập | `guided_choice` | — | Phân loại |
| Khớp regex | `guided_regex` | — | Ngày, mã, path |
| Grammar EBNF/Lark | `guided_grammar` | — | DSL, SQL con |
| JSON tự do | schema rỗng | `response_format={"type":"json_object"}` | Chỉ cần "là JSON" |

Backend mặc định từ vLLM v0.6.x là **`xgrammar`** (biên dịch grammar 1 lần rồi cache → overhead/token rất nhỏ). JSON schema đơn giản gần như không giảm throughput.

Đường sạch nhất qua LiteLLM — dùng `response_format json_schema`:
```python
resp = client.chat.completions.create(
    model="brain",
    messages=[...],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "plan", "schema": MY_SCHEMA}},
)
```

> **Đính chính về `drop_params: true`** (đang bật trong `config/litellm.yaml`): cơ chế thực tế là **`drop_params` chỉ lọc các param OpenAI-spec cấp cao mà provider không hỗ trợ**. Nội dung bên trong `extra_body` là **passthrough — LiteLLM forward thẳng vào request body và thường KHÔNG bị cắt**. Rủi ro drop chủ yếu xảy ra khi bạn truyền `guided_json` như **kwarg cấp cao** (không bọc trong `extra_body`). Vì vậy khuyến nghị **ưu tiên `response_format json_schema`** vẫn đúng (chuẩn OpenAI, portable), nhưng lý do không phải "extra_body bị cắt". Với các param riêng vLLM (`guided_regex/grammar/choice`), cứ bọc trong `extra_body`; nếu muốn chắc chắn tuyệt đối thì gọi thẳng `http://vllm:8000/v1`.

> **Đính chính về tool-calling "luôn hợp lệ"**: Cặp cờ `--enable-auto-tool-choice` + `--tool-call-parser hermes` là đúng và phù hợp Qwen2.5. NHƯNG với `tool_choice=auto` (chính là `--enable-auto-tool-choice`), model **sinh tự do rồi hermes parser TRÍCH** tool call — **không áp grammar**, nên JSON args của `read_file/write_file/list_dir/run_shell` **CÓ THỂ sai/thiếu, không được đảm bảo**. Muốn "luôn hợp lệ" phải dùng `tool_choice="required"` hoặc chỉ định named function kèm schema — lúc đó vLLM mới cưỡng chế bằng structured decoding. Đừng cho rằng auto = luôn đúng.

**Công sức: VỪA** (THẤP nếu chỉ `json_object`; CAO nếu viết grammar EBNF phức tạp).

#### 5) Tinh chỉnh Decoding params
Preset khuyến nghị cho code/agent:

| Param | Khuyến nghị `brain` |
|---|---|
| `temperature` | **0.0–0.2** (agent.py đang 0.2); 0 cho task cần lặp lại chính xác |
| `top_p` | 0.8–0.95 |
| `top_k` | 20–40 hoặc tắt (`-1`) khi đã dùng top_p |
| `min_p` | 0.05–0.1 (vLLM-specific) |
| `repetition_penalty` | 1.05–1.1 (hệ số **nhân**) chống lặp |
| `presence/frequency_penalty` | 0–0.3 (cộng, dải −2..2) |
| `max_tokens` | 1024–2048/lượt agent |
| `seed` | **cố định** khi debug → reproducible |

Lưu ý `repetition_penalty` (nhân, ~1.0–1.2) khác `presence/frequency_penalty` (cộng, −2..2). vLLM hỗ trợ cả ba — đừng chồng liều. **Công sức: THẤP.**

#### 6) Prefix Caching
Tái dùng **KV cache** phần đầu prompt trùng nhau giữa các request. Image mới chạy **V1 engine bật prefix caching MẶC ĐỊNH** → dù entrypoint không truyền cờ, tính năng vẫn hoạt động (thêm `--enable-prefix-caching` qua `VLLM_EXTRA_ARGS` để tường minh).
- **Không cần code**, chỉ cần giữ thứ tự prompt ổn định: bất biến (system → few-shot → context tĩnh) đặt **TRƯỚC**, động đặt **SAU**.
- Trên 3060 pool nhỏ hơn → cache dễ bị evict khi tải cao; server giữ cache lâu hơn.
- **Công sức: THẤP** (gần 0).

**Lộ trình nhóm A:** (1) chốt system prompt + decoding preset → (2) xác nhận prefix caching + sắp xếp prompt → (3) áp guided decoding cho output cấu trúc → (4) thêm few-shot tĩnh 3–5 ví dụ.

---

### 3.B — RAG (Retrieval-Augmented Generation): nạp tri thức KHÔNG train

RAG **không đụng một gram trọng số nào của `brain`**. Điểm mấu chốt: stack hiện tại **chỉ có model SINH, CHƯA có model EMBEDDING** → RAG bắt buộc thêm ít nhất 1 embedding model (thường + 1 vector store + 1 reranker).

Kiến trúc mục tiêu (self-host):
```
                       ┌─────────────────────────────────────────┐
  client → open-webui →│ LiteLLM (:4000)  brain / brain-pro / emb │
                       └───────┬───────────────┬────────────┬─────┘
                     vllm (:8000) brain   vllm-embed    reranker (vllm --task score
                     Qwen2.5-Coder-7B     bge-m3 (:8001)  hoặc /rerank)
                               │
                        Qdrant (:6333)  ← vector + sparse(BM25) + payload filter
```

Pipeline: `[OFFLINE] tài liệu → chunking → embedding → vector store (+BM25)` và `[ONLINE] câu hỏi → embedding → hybrid search → RRF → top-K → reranker → top-N → nhồi prompt → brain sinh`.

#### 1) Chunking
Cắt tài liệu theo **cấu trúc cú pháp** (hàm/class) cho code, không cắt mù theo ký tự. CPU thuần, vài giây → 1–2 phút cho cả codebase.
- Công cụ: LangChain `RecursiveCharacterTextSplitter`/`Language`-splitter, LlamaIndex `CodeSplitter`, hoặc **tree-sitter** (chính xác nhất). Tham số khởi điểm: `chunk_size ≈ 512–800 token`, `overlap ≈ 64–128`; với code: 1 hàm = 1 chunk + header (path + class) làm metadata.
- Codebase 3.000–5.000 file → ~20.000–40.000 chunk. Không cần nhãn. **Công sức: THẤP.**

#### 2) Embedding model (thêm vào stack — quan trọng nhất)

| Embedding model | Params | Dim | VRAM (FP16) | Ghi chú |
|---|---|---|---|---|
| **bge-m3** | 568M | 1024 | ~2–3 GB | Dense+sparse+ColBERT, đa ngữ (VN), ctx 8192. **Mặc định.** |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | ~1.5–2 GB | Cùng họ Qwen, nhẹ |
| Qwen3-Embedding-4B/8B | 4B/8B | 2560/4096 | ~9/~17 GB | Chỉ server ≥24GB |
| bge-small/base (ONNX/FastEmbed) | 33M/109M | 384/768 | **0 GB (CPU)** | Cho 3060 khi hết VRAM |

> **Vấn đề VRAM thực trên 3060:** `brain` với `GPU_MEMORY_UTILIZATION=0.90` **đã chiếm ~10.8GB**, chỉ còn ~1.2GB → **không đủ** nhét embedding GPU. Ba lựa chọn:
> 1. **Embedding trên CPU** bằng FastEmbed (ONNX, ~50–200 chunk/s) — **khuyến nghị cho 3060**.
> 2. Hạ `GPU_MEMORY_UTILIZATION` xuống ~0.78–0.80 rồi chạy vLLM embedding nhỏ (đổi lại KV cache co lại).
> 3. Giảm `MAX_MODEL_LEN` để lấy VRAM.
>
> **Server 24–48GB:** thoải mái — `brain` AWQ + KV cache + bge-m3 (~3GB) vẫn dư.

**Thời gian INDEXING** (không phải train): embed 20–40k chunk — GPU (bge-m3, batch 32–64) ~1–3 phút; CPU FastEmbed ~5–20 phút. Re-index tăng dần vài giây.

Serve: vLLM `--task embed` → `POST /v1/embeddings`; hoặc Infinity/TEI/FastEmbed. Khai báo trong `litellm.yaml`:
```yaml
  - model_name: brain-embed
    litellm_params:
      model: openai/bge-m3
      api_base: http://vllm-embed:8000/v1
      api_key: os.environ/VLLM_API_KEY
```
**Công sức: VỪA** (chủ yếu ops).

#### 3) Vector store
Lưu ~20–40k vector (1024-dim ≈ 2KB → ~40–80MB, rất nhỏ), chạy CPU/RAM, **không tốn VRAM**.

| Store | Hybrid sẵn | Filter | Hợp dự án? |
|---|---|---|---|
| **Qdrant** | Có (named sparse) | Mạnh | **Khuyến nghị chính** (hybrid+RRF built-in) |
| FAISS | Không | Không | Nhẹ nhất, POC (chỉ 1 file .index) |
| Chroma | Hạn chế | Có | Dễ xài, hơi nặng |
| pgvector | BM25 qua ParadeDB | SQL | Nếu đã bật Postgres cho LiteLLM |

**Công sức: THẤP–VỪA.**

#### 4) Hybrid search (BM25 + vector) + RRF
Kết hợp **BM25/sparse** (khớp tên hàm `parseInvoice` đúng chữ) + **dense** (khớp ngữ nghĩa "hàm tính thuế GTGT"). Với code, BM25 cực kỳ quan trọng. Hợp nhất bằng **RRF**: `score = Σ 1/(k+rank)`, k≈60.
- Qdrant hybrid: named vector `dense`+`sparse`, `FusionQuery(RRF)` trong 1 request. bge-m3 xuất sparse cùng dense → không tốn thêm model. **Công sức: VỪA.**

#### 5) Reranker
Cross-encoder đọc *cặp (câu hỏi, chunk)* → chấm precision cao hơn nhiều. Retrieve top-30..50 → rerank → giữ top-5..8 nhồi prompt.

| Reranker | Params | VRAM | Ghi chú |
|---|---|---|---|
| **bge-reranker-v2-m3** | 568M | ~2 GB | Đa ngữ (VN), chuẩn mực |
| bge-reranker-base | 278M | ~1.2 GB | Nhẹ hơn |
| Qwen3-Reranker-0.6B/4B | 0.6B/4B | ~2/9 GB | Cùng họ Qwen |

- **3060:** chạy reranker **CPU** (bge-reranker-base, 30–50 cặp ~0.3–1s) hoặc **bỏ ở dev, bật trên server**. **Server:** GPU thoải mái (~20–60ms/query).
- Serve: vLLM `--task score` → `POST /rerank` (tương thích Cohere/Jina). LiteLLM có `/rerank` endpoint. **Công sức: VỪA.**

#### 6) Agentic RAG / Retrieval-as-a-tool
Biến truy hồi thành **1 tool** `search_codebase(query, top_k)` để `brain` **tự quyết** khi nào tra, multi-hop. Rất khớp vì bạn đã có agent tool-calling hermes.
- Tự viết tool trong `agents/tools.py` (đúng phong cách repo). Handler: embed query → Qdrant hybrid → rerank → trả `{path, snippet, score}`.
- Cẩn thận **đa vòng retrieve** làm phình context trên 7B (chất lượng giảm khi context quá dài). **Công sức: VỪA → CAO.**

#### 7) RAG hay Fine-tune?

| Tiêu chí | Nghiêng **RAG** | Nghiêng **Fine-tune** |
|---|---|---|
| Bản chất cần thêm | **Tri thức/sự kiện** | **Hành vi/định dạng/văn phong** |
| Tần suất thay đổi | Đổi liên tục (mỗi commit) → re-index | Ổn định lâu dài |
| Trích dẫn nguồn / chống bịa | **Có** | Khó truy nguồn |
| Rủi ro | Retrieve sai → trả lời sai theo | Catastrophic forgetting |

**Kết luận cho dự án:** tri thức = codebase thay đổi liên tục → **RAG đúng**. Format tiếng Việt + kỷ luật tool-calling thì cần **LoRA mỏng**. Mô hình lý tưởng: **RAG cho tri thức + LoRA mỏng cho hành vi + `brain-pro` cho câu khó**.

**Lộ trình RAG:** MVP (tree-sitter → FastEmbed CPU → FAISS → tool dense) → Bản đủ (bge-m3 → Qdrant hybrid+RRF → bge-reranker-v2-m3) → Agentic.

---

### 3.C — PEFT: LoRA / QLoRA / DoRA (fine-tune nhẹ)

> **Cảnh báo nền tảng (đọc trước):** KHÔNG train PEFT trực tiếp trên bản AWQ — AWQ là định dạng **chỉ để inference**. Mọi phương pháp train trên bản gốc **bf16** `Qwen/Qwen2.5-Coder-7B-Instruct` (không hậu tố `-AWQ`), hoặc trên bản NF4 4-bit (QLoRA). Train xong mới tính chuyện serve.

Full fine-tune 7B cần **~16x bytes/param** (mục 3.D) → bất khả thi. PEFT đóng băng 99%+ trọng số gốc, chỉ train adapter vài chục–trăm MB.

#### 1) LoRA
`ΔW = B·A` rank thấp; forward `h = W·x + (α/r)·B·A·x`; chỉ `A,B` train, `W` đóng băng.
- **rank `r`**: 8–16 cho style/format; 32–64 cho task nặng. `alpha` thường = `2·r`. `dropout` 0.05–0.1. `target_modules` cho Qwen2.5: `q,k,v,o_proj, gate,up,down_proj` (đủ 7 lớp cho chất lượng tốt nhất).
- **Dữ liệu:** cặp instruction→response (ChatML). Style/tiếng Việt: **300–1.000** mẫu; task/tool-calling: **1.000–5.000**. Chất lượng > số lượng.
- **Khả thi 3060:** LoRA "thuần" (base bf16 ~14–15GB) **KHÔNG vừa 12GB** → phải dùng QLoRA. **Server 24–48GB:** LoRA 16-bit vừa (~18–26GB), ~1–2h cho 1–3k mẫu.
- Công cụ: **Unsloth** (số 1 cho 1 GPU, giảm ~50–70% VRAM, nhanh ~2x), TRL+PEFT, Axolotl/LLaMA-Factory.

#### 2) QLoRA — **ĐƯỜNG RAY CHÍNH CHO 3060**
LoRA + base nén **4-bit NF4** (bitsandbytes) + double quantization + paged optimizer.

| Tình huống train 7B QLoRA | VRAM | Ghi chú |
|---|---|---|
| Unsloth, seq 1024, batch 1 + grad-accum | **~5–7GB** | thoải mái trên 3060 |
| Unsloth, seq 2048, batch 1–2 | **~7–9GB** | vẫn vừa 12GB |
| TRL+PEFT (không Unsloth), seq 2048 | **~9–11GB** | sát trần |

- **Thời gian 3060:** 1.000 mẫu × 3 epoch, seq ~1024 → **~1–2h**; 5.000 mẫu → **~4–8h**. **Server:** ~20–40 phút cho 1–3k mẫu (thường không cần QLoRA, dùng LoRA 16-bit chất lượng nhỉnh hơn).
- Ưu: **train được 7B trên 12GB**, chất lượng gần LoRA 16-bit. Nhược: forward chậm hơn ~20–30% do dequant.

> **Đính chính công cụ:** NF4 **KHÔNG cần** bitsandbytes 0.44 — NF4 đã có từ ~bnb 0.39/0.40 (2023, thời QLoRA). `>=0.44` chỉ là mức sàn khuyến nghị gần đây, không phải điều kiện để có NF4. Các phần còn lại (Unsloth giảm 50–70% VRAM; `use_dora`/`use_rslora` trong PEFT; `peft>=0.12–0.13`) chính xác.

#### 3) DoRA / QDoRA
Tách trọng số thành **magnitude** + **direction**: `W = m·(V/||V||)`; train `m` riêng + LoRA cho hướng. Nhỉnh hơn LoRA vài điểm ở **rank thấp (r=4–8)**, hữu ích khi data **ít** (≤~1.000 mẫu).
- 3060: QDoRA ~6–9GB, chậm hơn LoRA/QLoRA ~1.2–2x. PEFT `LoraConfig(use_dora=True)`.
- ⚠️ Hỗ trợ DoRA-adapter runtime trong vLLM **kém chín hơn LoRA thuần** → **khuyến nghị merge DoRA vào base rồi serve**, không hot-swap.

#### 4) LoRA+ và rsLoRA (bật cờ, không đổi cách serve)

| Biến thể | Thay đổi | Khi nào dùng | Chi phí |
|---|---|---|---|
| **LoRA+** | LR(B) > LR(A) (8–16x) | hầu như luôn nên bật | ~0 |
| **rsLoRA** | scale `α/√r` thay `α/r` | khi rank ≥ 64 | ~0 |

#### Serve PEFT vào stack — **NÚT THẮT AWQ**

vLLM serve LoRA runtime:
```
--enable-lora --max-lora-rank 16 --max-loras 4 \
--lora-modules brain-vi=/models/adapters/brain-vi
```
Client gửi `model="brain-vi"` → áp adapter; `model="brain"` → base. Hot-swap qua `POST /v1/load_lora_adapter` (bật `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`).

> **Đính chính quan trọng (nút thắt AWQ):** Adapter được fit trên trọng số **bf16** nhưng phục vụ trên base mang sẵn sai số lượng tử hóa **AWQ** — độ lệch **không nhất thiết "nhẹ"**. Hơn nữa vLLM hỗ trợ hot-swap LoRA **chồng lên base AWQ là hạn chế/phụ thuộc method** (khả năng và chất lượng không đảm bảo). **Hai hướng KHÔNG ngang nhau:**
> - **(A) Hot-swap trên base AWQ** — đường **yếu nhất**, phải test kỹ trên đúng version vLLM và đo lại chất lượng. Vừa 12GB, linh hoạt nhưng rủi ro.
> - **(B) Merge → re-quantize AWQ** — **đường tin cậy**: `merge_and_unload()` ra bf16 → AutoAWQ quantize → serve như `brain`. Mất hot-swap, nhưng khớp precision nhất. **Khuyến nghị cho 3060 nếu chỉ 1 adapter cố định.**
> - Trên **server 24–48GB**: serve thẳng **base bf16 + LoRA hot-swap** (đúng precision đã train, hết mismatch).
>
> Ràng buộc: `--max-lora-rank` **phải ≥ r** đã train. Sau fine-tune, **regression test tool-calling** để chắc adapter không phá format `hermes`.

---

### 3.D — Full SFT toàn phần & Continued Pre-training (fine-tune nặng)

**Gần như LOẠI TRỪ trên phần cứng của bạn** — trình bày để bạn biết vì sao không đi đường này.

Full fine-tune 7B với mixed-precision AdamW cần lưu đồng thời, **cho mỗi tham số**:
- 2 bytes bf16 weight + 2 bytes bf16 gradient + **12 bytes fp32 optimizer** (master copy fp32 + momentum `m` fp32 + variance `v` fp32) = **16 bytes/param**.

> **Đính chính số học:** "fp32 master + m + v" là **3 tensor fp32 = 12 bytes/param ≈ 84GB** chỉ riêng optimizer states cho 7B (KHÔNG phải 56GB — con số 56GB chỉ ứng với m+v mà thiếu bản master fp32). Cộng weight+grad → **~16 bytes/param ≈ 112GB cho 7B**; Qwen2.5-7B thực ~7.6B params → **~122GB** chỉ riêng model states, **chưa tính** activation/KV. Con số này **cao hơn** mọi ước lượng "84–90GB" và khẳng định: **bất khả thi trên 3060 12GB, và cả trên server 24–48GB đơn GPU** (cần FSDP/multi-GPU hoặc offload nặng).

Continued Pre-training (CPT — train tiếp trên corpus lớn không nhãn để nạp domain/ngôn ngữ) có cùng profile bộ nhớ như full-FT, thêm rủi ro **catastrophic forgetting** rất cao. Với self-host gọn nhẹ, **không đi đường này** — dùng RAG cho tri thức, QLoRA cho hành vi.

**Sản phẩm cuối** (nếu vẫn làm trên cụm GPU): checkpoint **BF16 merged**.

> **Đính chính về serving:** re-quant về AWQ là **TÙY CHỌN, không bắt buộc** — chỉ cần khi phải nhét model vào VRAM nhỏ. `GPU_MEMORY_UTILIZATION` và `MAX_MODEL_LEN` là **cờ runtime của vLLM, hoàn toàn độc lập với định dạng quantization**; chúng vẫn dùng được khi serve BF16 nguyên bản (nếu đủ VRAM), hoặc GPTQ/FP8/bitsandbytes. Việc "giữ đúng 2 setting này" **không hề đòi hỏi AWQ**.

---

### 3.E — Preference Optimization: căn chỉnh hành vi SAU SFT

> **3 sự thật chi phối:** (1) KHÔNG train trên AWQ — train trên base bf16, tạo LoRA, rồi đưa lại pipeline. (2) Đây là bước **SAU SFT**, không thay SFT (ngoại lệ: **ORPO** gộp cả hai). (3) Phải giữ nguyên **chat template + format tool-call `hermes`** — nếu data preference không dùng đúng chat template Qwen2.5 (kể cả block `<tool_call>`), bạn "align" khẩu vị mới nhưng **làm hỏng tool-calling** — rủi ro số 1.

Tất cả cùng giải: *"model nên thích A hơn B"*. Trục nặng→nhẹ hạ tầng: `PPO ≫ DPO ≈ IPO > KTO > SimPO ≈ ORPO`.

#### RLHF-PPO — **không khuyến nghị**
Giữ **4 model đồng thời** (policy + reference + reward + critic). Cần **10k–50k+** comparison cho RM.
- **3060: KHÔNG khả thi** (OOM ngay). **Server 24–48GB:** biên, mong manh, không nên.
- **Công sức: CAO.** Chỉ dùng khi có cụm GPU + đội lớn.

#### DPO — **khuyến nghị mặc định**
Bỏ RM và bỏ vòng RL; tối ưu binary cross-entropy trên cặp `(chosen, rejected)`, so tương đối với reference. Với LoRA, reference = **base đóng băng** (TRL disable-adapter, KHÔNG copy model thứ 2 vào VRAM).
- Data: `{prompt, chosen, rejected}`. Domain hẹp: **500–3.000 cặp** đã khác biệt rõ. Nguồn: `brain-pro` (Claude) tạo `chosen`, output `brain` cũ làm `rejected`.

| Cấu hình | RTX 3060 12GB | Server 24–48GB |
|---|---|---|
| **QLoRA-DPO 7B** (4-bit, seq 1024, bs=1, grad-ckpt) | **~9–11GB (tight)**, ~2–3k cặp: **3–6h** | ~14–18GB, **~1–2h** |
| LoRA-DPO 7B (bf16) | Không (base ~15GB vượt) | ~22–30GB, ~1–2h |

> **Trả lời "LoRA-DPO trên 3060 khả thi?": CÓ, nhưng phải là QLoRA-DPO** (base 4-bit), seq ≤ 1024, batch=1 + grad-accum + grad-checkpoint, dùng **Unsloth**. Không phải LoRA bf16 thuần.

> **Đính chính cỡ model lớn:** con số "QLoRA-DPO 14B ~20–30GB, 32B ~48GB" và việc "khớp ngưỡng auto-detect" là **lẫn hai loại bộ nhớ**. Ngưỡng `detect-gpu.sh` (≥24GB→32B-AWQ, ≥64GB→32B full) là để **CHỌN MODEL PHỤC VỤ (serving)**, **không phải** bộ nhớ **QLoRA-DPO training** — không thể "khớp" hai thứ. Thực tế QLoRA-DPO 14B thường **~12–18GB** (chỉ lên 20–30GB khi seq/batch lớn). Ngoài ra 32B **bf16 full nặng ~64GB CHỈ riêng trọng số**; cộng KV cache + activation thì 1 GPU 64GB **không đủ/biên gắt** (thực tế cần ~80GB như H100-80GB hoặc multi-GPU) — mốc "≥64GB chọn 32B full" quá lạc quan cho serving.

- Công cụ: TRL `DPOTrainer` + PEFT + bitsandbytes + Unsloth. `loss_type="sigmoid"` mặc định. **Công sức: VỪA.**

#### ORPO — **hợp nhất cho setup nhỏ**
Gộp SFT + preference 1 bước, **bỏ reference model**. Loss = NLL(chosen) + λ·odds-ratio penalty.
- Tiết kiệm ~15–30% VRAM so DPO. **3060: khả thi** QLoRA-ORPO 7B (~8–10GB), ~2–5h. Data: **1k–10k cặp**.
- Phù hợp khi **chưa SFT domain** (một công đôi việc). TRL `ORPOTrainer`. **Công sức: THẤP–VỪA.**

#### KTO — **khi không có cặp**
Chỉ cần nhãn nhị phân **tốt/xấu** cho từng mẫu rời (`{prompt, completion, label: true/false}`), không cần ghép cặp.
- **Rất hợp thu tín hiệu production:** thumbs 👍/👎 từ Open WebUI, "test pass = tốt / fail = xấu" từ agent → dataset gần như tự động. Chịu mất cân bằng (`desirable_weight`).
- 3060: QLoRA-KTO 7B ~9–11GB, ~3–6h. TRL `KTOTrainer`. **Công sức: VỪA.**

#### SimPO & IPO (biến thể DPO)
- **SimPO:** bỏ reference, reward chuẩn hóa theo độ dài + margin `γ` → chống length-bias, nhẹ. 3060 khả thi (~8–10GB). TRL `CPOTrainer(loss_type="simpo")`. Rất nhạy hyperparam.
- **IPO:** sửa overfitting của DPO (regularize), giữ reference. TRL `DPOTrainer(loss_type="ipo")`. Ổn định hơn khi data ít/nhiễu.

**Bảng chọn theo tình huống:**

| Bạn đang có... | Nên dùng |
|---|---|
| Chưa SFT domain, muốn 1 bước | **ORPO** |
| Đã có cặp chất lượng (Claude tạo `chosen`) | **DPO** (fallback **IPO** nếu data ít/nhiễu) |
| Chỉ có log 👍/👎 hoặc pass/fail test | **KTO** |
| Muốn nhẹ RAM + chống câu dài lan man | **SimPO** |

**Đường khả thi nhất:** thu vài nghìn cặp bằng `brain-pro` sinh `chosen` + output `brain` cũ làm `rejected` (đúng chat template Qwen2.5 kèm tool-call) → **QLoRA-ORPO hoặc QLoRA-DPO 7B bằng Unsloth** (train ngay trên 3060, 3–6h) → merge → AutoAWQ re-quant → giữ `SERVED_MODEL_NAME=brain` → kiểm thử tool-call hermes trước khi chốt.

---

### 3.F — Distillation · Model Merging · Quantization-for-serving · Speculative Decoding

Tầng **tối ưu & tổng hợp**. Ba trong bốn (merging, quantization, speculative n-gram) gần như **không cần training data**.

#### 1) Knowledge Distillation
Chuyển tri thức từ teacher (`brain-pro` = Claude) sang student (Qwen 7B). Vì teacher là **API kín**:

| Biến thể | Dùng được với Claude? |
|---|---|
| **Off-policy / sequence-level KD** (teacher sinh cặp → student SFT) | **CÓ** — chỉ cần text. Rẻ nhất, khả thi nhất |
| **On-policy KD** (student sinh → teacher critique-and-revise) | CÓ, dạng hạn chế (chỉ mức text, không logits) |
| **Logit-level / white-box KD** (KL trên token distributions) | **KHÔNG với Claude** — cần logits (chỉ làm được với teacher open như Qwen2.5-Coder-32B) |

- Data: prompt từ codebase riêng + **tool-calling traces** đúng hermes. Domain hẹp ~3k–20k mẫu SFT.
- Train student = LoRA/QLoRA (giống 3.C): 3060 QLoRA ~9–11GB (chậm, 10k mẫu ~nhiều giờ→1 ngày); server ~1–4h.
- Công cụ: TRL (`SFTTrainer`; `GKDTrainer` khi có teacher open), PEFT, bitsandbytes, **Unsloth**. Data-gen gọi Claude qua LiteLLM `brain-pro`.
- Serve: train LoRA trên base FP16 → merge → **llm-compressor → AWQ** → đổi `MODEL`. **Công sức: vừa → cao.**

#### 2) Model Merging (mergekit)
Gộp trọng số nhiều model **cùng kiến trúc + cùng tokenizer** bằng số học tensor — **không train**.

| Method | Cơ chế |
|---|---|
| Linear / Model Soup | Trung bình có trọng số |
| SLERP | Nội suy cầu giữa 2 model |
| Task Arithmetic | `task_vector = finetuned − base`; cộng/trừ kỹ năng |
| **TIES** | Trim + elect sign + merge → giảm interference |
| **DARE(-TIES)** | Drop-And-REscale trước merge |

- Chạy CPU/RAM (~16–32GB RAM), vài phút → ~20 phút, **gần như không cần GPU**. **KHÔNG merge được bản AWQ**, không cross-family.
- Output FP16 → phải **quant → AWQ** để serve trên 3060. Use case: gộp coder + model Qwen tiếng Việt để cải thiện output tiếng Việt.
- ⚠️ **Bắt buộc eval tool-calling** — merge sai dễ hỏng format hermes. **Công sức: thấp → vừa.**

#### 3) Quantization-for-serving (llm-compressor)
PTQ nén model đã tune/merge (FP16) xuống 4/8-bit để serve. Một lượt **calibration forward-only** (không train).

| Format | Bit | Hợp Ampere (RTX 3060)? |
|---|---|---|
| **AWQ** | 4-bit W | ✅ (`awq_marlin`) — đang dùng |
| **GPTQ** | 4-bit W | ✅ (`gptq_marlin`) |
| INT8 W8A8 (SmoothQuant) | 8-bit | ✅ gần lossless |
| **FP8** | 8-bit | ❌ **không tăng tốc** (cần sm_89+ Ada/Hopper) |
| GGUF | 4–8 | vLLM experimental → bỏ qua |

- Calibration set **128–512 mẫu** dùng **chính data domain** (code + tool-calling tiếng Việt) → quant khớp use case.
- Công cụ: **`llm-compressor`** (chính chủ vLLM, thay AutoAWQ đã deprecated) → xuất compressed-tensors load thẳng vLLM. Set `MODEL` + `QUANTIZATION` trong `.env`.

> **Đính chính VRAM khi quantize:** quant AWQ/GPTQ diễn ra **TUẦN TỰ theo từng decoder layer** (sequential/layer-by-layer), peak VRAM chỉ ~1 layer + activation calibrate (~**4–8GB**), **KHÔNG cần cả 15GB nằm trên GPU cùng lúc** — bản fp16 để ở CPU RAM. Vì vậy **AWQ-quantize 1 model 7B HOÀN TOÀN khả thi trên card 12GB** (chỉ chậm hơn), không phải "gần như không khả thi". Khuyến nghị dùng server lớn (~15–40 phút với 128–512 mẫu) chỉ để **nhanh hơn**, không phải vì bất khả thi. **Công sức: thấp → vừa.**

#### 4) Speculative Decoding
Model **draft** sinh nhanh `k` token, **target** (`brain`) verify cả `k` trong 1 forward → tăng tốc **lossless** (output y hệt).

| Kiểu | Cần train? | VRAM thêm | 3060 |
|---|---|---|---|
| **N-gram / prompt-lookup** | ❌ | ~0 | ✅ **tốt nhất** — rất hợp code (lặp nhiều) |
| Draft model (Qwen2.5-Coder-0.5B) | ❌ | ~1–1.5GB | ⚠️ có thể phải giảm `MAX_MODEL_LEN` 32768→~16384 |
| EAGLE-3 / Medusa | ✅ (train head) | nhỏ–vừa | ⚠️ chật |

- **Speedup ước lượng:** n-gram ~1.3–2×; draft ~1.5–2.5×; EAGLE-3 ~2–4×. Lợi nhất ở **batch thấp/latency-bound** (đúng self-host 1–vài user).
- Cắm qua `VLLM_EXTRA_ARGS`:
  ```
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
  ```
- Target AWQ + speculative: **hỗ trợ** trong vLLM V1. Draft phải cùng tokenizer family. **Công sức: thấp (n-gram) → cao (EAGLE).**

**Thứ tự áp dụng nhóm F:** (1) Speculative n-gram (0 VRAM, 1 dòng env) → (2) Distillation off-policy từ `brain-pro` → (3) Merging (nếu có ≥2 checkpoint) → (4) Re-quantize (llm-compressor, AWQ) trên server.

---

## 4. Bảng tổng so sánh mọi phương pháp

| Phương pháp | Đụng trọng số? | VRAM trên 3060 12GB | Dữ liệu cần | Công sức | Khi nào dùng |
|---|---|---|---|---|---|
| System prompt | Không | ~0 (KV) | 1 văn bản | Thấp | Luôn — nền tảng |
| Few-shot / ICL | Không | ~0 (KV) | 3–5 ví dụ | Thấp–Vừa | Ép format/style nhanh |
| Output steering | Không | ~0 | 0 | Thấp–Vừa | Cấm/ưu tiên token, cắt |
| **Guided decoding** | Không | ~0 | 1 schema/grammar | Vừa | **Output cấu trúc agent (đòn bẩy lớn)** |
| Decoding params | Không | ~0 | 0 (cần A/B) | Thấp | Ổn định, chống lặp, reproducible |
| Prefix caching | Không | ~0 | 0 | Thấp | Agent nhiều lượt (tốc độ) |
| **RAG** | Không | CPU embed / ~2–3GB nếu GPU | Chunk codebase (không nhãn) | Vừa | **Nạp tri thức thay đổi liên tục** |
| LoRA (16-bit) | Train adapter | ❌ base bf16 ~15GB không vừa | 300–5.000 cặp | Thấp–Vừa | Trên server; hành vi/format |
| **QLoRA** | Train adapter | ✅ **~5–9GB** | 300–5.000 cặp | Thấp–Vừa | **Đường ray SFT chính cho 3060** |
| DoRA / QDoRA | Train adapter | ✅ ~6–9GB | ≤~1.000 cặp | Vừa | Data ít, ép chất ở rank thấp |
| **DPO** | Train adapter | ✅ QLoRA-DPO ~9–11GB | 500–3.000 cặp | Vừa | Đã có cặp; căn chỉnh khẩu vị |
| **ORPO** | Train adapter | ✅ ~8–10GB | 1k–10k cặp | Thấp–Vừa | Gộp SFT+preference, chưa SFT |
| KTO | Train adapter | ✅ ~9–11GB | 1k–10k nhãn 👍/👎 | Vừa | Chỉ có log pass/fail, thumbs |
| SimPO / IPO | Train adapter | ✅ ~8–11GB | 1k–10k cặp | Vừa | SimPO nhẹ/anti-length; IPO ổn định |
| PPO | Train 2 lần | ❌ OOM (4 model) | 10k–50k+ ranking | Cao | Không khuyến nghị self-host nhỏ |
| **Full-SFT / CPT** | Train toàn bộ | ❌ **~112–122GB, bất khả thi** | 10k+ | Cao | Loại trừ trên phần cứng của bạn |
| Distillation | Train student | ✅ QLoRA ~9–11GB (chậm) | 3k–20k synthetic | Vừa–Cao | Chuyển tri thức Claude→Qwen |
| Model Merging | Đụng weights, không train | ✅ CPU/RAM vài phút | 0 (chỉ eval) | Thấp–Vừa | Gộp kỹ năng cùng-family |
| Quantization-serving | Biến đổi weights, không train | ✅ quantize ~4–8GB peak; serve ✅ | calib 128–512 | Thấp–Vừa | Nén model đã tune để serve |
| Speculative (n-gram) | Không | ✅ ~0 | 0 | Thấp | Tăng tốc lossless ngay |

---

## 5. Chuẩn bị Dữ liệu & Đánh giá — nhóm hay bị bỏ qua nhất

> Phần này chi phối **80% kết quả** nhưng hay bị xem nhẹ. LoRA train trên dataset bẩn/leak sẽ *trông* giỏi trên eval nhưng hỏng trong production. Dataset sạch + eval trung thực cho phép train ít (500–2.000 mẫu) mà vẫn nâng chất lượng thật. **Toàn bộ phần này chạy được trên 3060, không đòi VRAM lớn** — chỉ cần đường serve vLLM sẵn có + token API `brain-pro`.

### Phần A — Xây dựng Dataset

**A1. Nguồn dữ liệu:**

| Nguồn | Chất lượng | Rủi ro |
|---|---|---|
| **Log thật (production traces)** | Phân phối *đúng* use case | Chứa lỗi model yếu; PII/secret |
| **Distill từ `brain-pro`** | Rất cao, sát domain | License output; vẫn có hallucinate |
| **Synthetic (self-instruct)** | Rộng, rẻ | Kém đa dạng thật |
| **Con người** | Cao nhất | Ít, tốn công |

Với use case của bạn: **ưu tiên distill từ `brain-pro`** (mục tiêu **1.000–3.000 cặp** đủ cho LoRA đổi style/format rõ rệt) + log thật (lọc phiên *thất bại* làm hard negatives) + 50–200 mẫu human cho edge-case. Công cụ: gọi LiteLLM `model="brain-pro"`; `distilabel`; LiteLLM **success_callback** ghi trace ra JSONL. Không cần GPU.

**A2. Clean / Dedup / Decontamination:**
- **Cleaning:** bỏ mẫu hỏng (JSON tool-call không parse, trả lời tiếng Anh khi cần tiếng Việt), scrub secret (`detect-secrets`, `gitleaks`).
- **Dedup:** exact hash + near-dup (MinHash qua `datasketch`/`text-dedup`; semantic cosine >0.95 qua `sentence-transformers`).
- **Decontamination (quan trọng nhất):** loại khỏi train mọi mẫu trùng **13-gram** với test set benchmark (HumanEval/MBPP) và test split riêng. Không làm = điểm eval là **giả**.
- Sau clean+dedup thường rụng 15–40%. Chạy CPU. **Công sức: Vừa.**

**A3. Format theo Chat Template Qwen2.5 (ChatML) — RẤT quan trọng:**
```
<|im_start|>system
{system + tool definitions}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{trả lời, hoặc <tool_call>\n{"name":...,"arguments":{...}}\n</tool_call>}<|im_end|>
```
- Special tokens `<|im_start|>`, `<|im_end|>` là **token thật** trong vocab. Tool-call `<tool_call>...</tool_call>` **chính** là định dạng parser `hermes` bóc.
- **Nguyên tắc vàng: template lúc train = template lúc serve.** Dùng `tokenizer.apply_chat_template(messages, tools=..., tokenize=False)` — **tuyệt đối không tự viết tay chuỗi ChatML**.
- Chỉ tính **loss trên phần assistant** (mask system/user) — "train on completions only". **Công sức: Thấp–Vừa.**

**A4. Lượng data cần:**

| Mục tiêu | Phương pháp | Số mẫu |
|---|---|---|
| Đổi style/format (giọng tiếng Việt) | LoRA SFT | 300–1.500 |
| Format tool-call ổn định | LoRA SFT (+ traces) | 500–2.000 |
| Domain adaptation (API riêng) | LoRA/QLoRA SFT | 2.000–10.000 |
| Preference tuning | DPO/ORPO | 1.000–10.000 cặp |
| Hành vi sâu rộng | Full SFT | 10.000+ (thường không cần) |

Quy tắc an toàn: 1–3 epoch, thà nhiều data 1 epoch hơn ít data 5 epoch (LoRA rank thấp overfit rất nhanh).

**A5. Split:** 80/10/10 hoặc 90/5/5. Quan trọng hơn tỉ lệ: **test là hold-out theo thời gian/task** (hash theo *task id*, không random), tránh leakage. Giữ **20–50 kịch bản** agent làm regression suite (tách khỏi train).

### Phần B — Đánh giá

> **Đo trước, train sau, đo lại.** Luôn có **baseline = `brain` gốc**. Chạy eval **qua đúng đường serve thật** (`base_url = LiteLLM`, `model = "brain"`) để đo chính xác cái user gặp (kể cả ảnh hưởng AWQ, chat template, sampling).

**B1. Benchmark code chuẩn:** HumanEval (**164 bài**), MBPP (full **974**, sanitized **427**), **EvalPlus** (HumanEval+/MBPP+ thêm ~80x test → điểm sát thực tế). Chỉ đo năng lực code chung, **không** phủ domain/tiếng Việt/agent. HumanEval qua vLLM ~5–20 phút trên 3060. Chạy sinh code trong **sandbox**.

> **Đính chính công cụ:** **EvalPlus** có backend OpenAI-compatible native (`evalplus.evaluate --backend openai --base-url ...`) → **trỏ thẳng vLLM/LiteLLM OK**. Nhưng **bigcode-evaluation-harness** được thiết kế để sinh code cục bộ qua **HF transformers/accelerate**, **KHÔNG có runner endpoint OpenAI-compatible first-class** — muốn chấm qua vLLM thường dùng EvalPlus, không "trỏ thẳng endpoint" như EvalPlus. **Công sức: Thấp.**

**B2. LLM-as-Judge (`brain-pro` làm giám khảo):** đưa câu trả lời `brain` cho Claude chấm theo rubric hoặc so cặp (pairwise). Eval set **50–200 prompt** phủ code/sửa bug/giải thích/agent. Công cụ: **`promptfoo`** (provider `brain` + judge `brain-pro`, assertion `llm-rubric`), **DeepEval** (G-Eval). Judge = Claude (khác họ với Qwen student) → **ít self-enhancement bias**. Random hóa thứ tự A/B để chống bias vị trí. **Công sức: Vừa.**

**B3. Task-specific eval:** 30–150 case domain riêng, assert bằng luật (regex, JSON schema, compile được, ngôn ngữ = vi qua `fasttext-langdetect`). Là **cổng CI** trước khi promote LoRA. **Công sức: Vừa–Cao.**

**B4. Regression test agent tool-calling — RỦI RO SỐ 1:** 20–50 kịch bản golden, replay qua `brain`, kiểm: `<tool_call>` parser `hermes` bóc thành công? đúng tool? arguments đúng schema (`jsonschema`)? multi-step kết thúc đúng? Gồm cả case "**không được gọi tool**" để bắt over-calling. Test **qua đúng đường vLLM có `--tool-call-parser hermes`**, `temperature=0`, `run_shell` trong sandbox. **MUST-HAVE CI gate:** LoRA làm rớt tool-calling → **không deploy** dù HumanEval tăng. **Công sức: Cao (đáng nhất).**

**B5. Overfit / Leakage detection:** theo dõi val loss (quay đầu tăng = overfit); held-out gap (val cao/test thấp = leakage); n-gram overlap (13-gram) train↔benchmark; canary/memorization probe (model hoàn thành *nguyên văn* train sample = học vẹt). Công cụ: `wandb`/`tensorboard`, `datasketch`. Là **cổng cuối** trước khi quant AWQ. **Công sức: Vừa.**

---

## 6. Đưa model đã tinh chỉnh vào chính platform này

Một adapter/model đã tune xong LANDING vào stack vLLM + LiteLLM ra sao. 3 phương pháp con: **(A)** LoRA động, **(B)** merge + re-quant, **(C)** vận hành (A/B, rollback, versioning, VRAM).

| Tiêu chí | (A) LoRA động | (B) Merge + re-quant AWQ |
|---|---|---|
| Đụng trọng số base? | Không (patch adapter runtime) | Có (fuse → model mới) |
| Nhiều phiên bản cùng lúc? | Rất tốt (nhiều adapter/1 base) | Kém (mỗi model 1 bộ weights) |
| Latency | +~5–15% | Bằng model gốc (nhanh nhất) |
| Vừa 3060 12GB? | Có (base AWQ + max-loras nhỏ) | Serve: có; re-quant nên làm trên server |
| Rollback | Hot-swap không restart | Đổi `MODEL` + restart |

**Nguyên tắc:** mặc định **(A)** để lặp nhanh + A/B rẻ; chuyển **(B)** khi adapter đã chốt và cần latency tối đa.

### (A) Serve LoRA động
```bash
# .env — vllm-entrypoint.sh đã hỗ trợ VLLM_EXTRA_ARGS, KHÔNG cần sửa code
VLLM_EXTRA_ARGS=--enable-prefix-caching --enable-lora --max-loras 2 --max-lora-rank 16 --lora-modules brain-vi=/adapters/brain-vi
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
```
Mount `- ./adapters:/adapters:ro` trong `docker-compose.yml`. Đăng ký LiteLLM:
```yaml
  - model_name: brain-vi
    litellm_params:
      model: openai/brain-vi          # trùng tên trong --lora-modules
      api_base: http://vllm:8000/v1
      api_key: os.environ/VLLM_API_KEY
```
Hot-swap version không restart:
```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name":"brain-vi","lora_path":"/adapters/brain-vi-v2"}'
```
VRAM 3060: base ~4.5–5GB + buffer ~0.5–1GB + ~50–300MB/adapter → còn ~5–6GB KV (32k vẫn ổn nếu `--max-loras 1–2`; hạ `MAX_MODEL_LEN` 16k nếu ≥2 adapter). **Test `--enable-lora` trên base AWQ NGAY từ đầu** — nếu vLLM báo không hỗ trợ LoRA với `awq_marlin`, đi hẳn (B). **Công sức: Thấp–Vừa.**

### (B) Merge + re-quantize AWQ
```
Qwen2.5-Coder-7B-Instruct (fp16) + adapter (PEFT)
   │ merge_and_unload()   ← merge trên bf16, KHÔNG trên AWQ
   ▼
brain-vi-merged-fp16 ──AutoAWQ/llm-compressor──► brain-vi-awq
```
```bash
# .env
MODEL_AUTODETECT=false          # BẮT BUỘC — nếu không, detect-gpu.sh ghi đè MODEL về Qwen gốc mỗi restart
MODEL=/models/brain-vi-awq
SERVED_MODEL_NAME=brain-vi
QUANTIZATION=awq_marlin
MAX_MODEL_LEN=32768
```
Merge chạy CPU/RAM (~32GB) nếu không đủ GPU. Re-quant khả thi ngay trên 3060 (~4–8GB peak, layer-by-layer) nhưng làm trên server nhanh hơn (~15–40 phút). Hạn chế: `.env` hiện nuôi **1 container = 1 model** — muốn chạy `brain` + `brain-vi` song song phải thêm service vLLM thứ hai (tốn VRAM). **Công sức: Vừa–Cao.**

### (C) A/B, rollback, versioning, quản VRAM
- **A/B qua gateway:** nhiều deployment cùng `model_name` với `weight` khác nhau (vd 90/10); client vẫn gọi `model="brain"`. Bật `database_url` để log usage/latency. Fallback `fallbacks: [{"brain": ["brain-pro"]}]`.
- **Rollback:** (A) runtime unload/load ~0 downtime; gateway hạ `weight`→0 ~0 downtime; (B) đổi `MODEL` + restart vài phút. **Luôn giữ nút gateway làm lớp rollback tức thời.**
- **Versioning:** thư mục có version (`adapters/brain-vi/v2`), `SERVED_MODEL_NAME=brain-vi-v2` (tên có version) → LiteLLM map tên client ổn định `brain-vi`. Commit `litellm.yaml` + `.env` làm nguồn sự thật.
- **VRAM nhiều adapter (3060):** `--max-loras 1–2`, `--max-cpu-loras 4–8` (giữ nhiều version trên RAM, swap khi cần, gần như miễn phí VRAM), `--max-lora-rank 16` (đừng đặt dư).

**File cần đụng khi land:** `config/litellm.yaml` (thêm `model_name`), `.env` (`VLLM_EXTRA_ARGS`/`MODEL`/`SERVED_MODEL_NAME`/`MODEL_AUTODETECT`), `docker-compose.yml` (mount `./adapters` hoặc `./models`). `scripts/vllm-entrypoint.sh` đã hỗ trợ `VLLM_EXTRA_ARGS` + `QUANTIZATION` → **không cần sửa** cho phần lớn trường hợp.

---

## 7. Kết: lộ trình đề xuất & cạm bẫy

### Lộ trình 4 bước cụ thể cho bạn

1. **Khai thác cạn tầng KHÔNG train (1–2 buổi, chạy ngay trên 3060):** chốt system prompt + decoding preset (`temperature=0.2, top_p=0.9, repetition_penalty~1.07, seed` khi debug) → xác nhận prefix caching + sắp xếp prompt bất biến lên đầu → **áp guided decoding `response_format json_schema`** cho output cấu trúc agent (cú hích tin cậy lớn nhất) → thêm few-shot tĩnh.

2. **Dựng RAG cho tri thức codebase (không train brain):** MVP tree-sitter chunking → FastEmbed CPU → FAISS → tool `search_codebase`; nâng lên bge-m3 + Qdrant hybrid+RRF + bge-reranker khi cần. Đây là cách đúng cho tri thức thay đổi liên tục.

3. **Chỉ khi vẫn thiếu HÀNH VI/ĐỊNH DẠNG bền vững → QLoRA-SFT bằng Unsloth** (bật LoRA+, `r=16, alpha=32, dropout=0.05`, đủ 7 module) trên base **bf16** (KHÔNG `-AWQ`), data 500–3.000 cặp ChatML tiếng Việt **kèm tool-call hermes**. Train ngay trên 3060 (~5–9GB, 1–3h). Nếu cần căn chỉnh khẩu vị: **QLoRA-DPO/ORPO**.

4. **Land vào platform:** ưu tiên **(A) LoRA động** (A/B rẻ, hot-swap); nếu LoRA-over-AWQ không ổn định trên version vLLM của bạn → **(B) merge → re-quant AWQ** (đường tin cậy nhất), chạy re-quant trên server rồi bê file về serve. Luôn giữ `MODEL_AUTODETECT=false` khi serve model tự làm.

Xuyên suốt: **đo baseline TRƯỚC khi train** (B1 EvalPlus + B2 judge + **B4 agent regression**), giữ hybrid `brain-pro` cho câu khó.

### Cạm bẫy phải tránh (checklist)

- ⚠️ **Train nhầm trên AWQ:** luôn train trên base **bf16** `Qwen/Qwen2.5-Coder-7B-Instruct`, không phải bản `-AWQ` (AWQ inference-only, không có gradient path).
- ⚠️ **Break tool-calling:** không giữ đúng chat template + `<tool_call>` trong data → agent hỏng âm thầm. **B4 regression là CI gate bắt buộc.** Nhớ: `tool_choice=auto` **không** đảm bảo JSON args hợp lệ — muốn cưỡng chế phải `required` + schema.
- ⚠️ **Nút thắt AWQ khi serve LoRA:** adapter bf16 trên base AWQ **không mismatch "nhẹ"** — đường (A) hot-swap là yếu nhất; đường tin cậy là serve trên base bf16 (server) hoặc merge→re-quant (B).
- ⚠️ **Catastrophic forgetting:** không CPT/full-FT trên phần cứng của bạn (cần ~112–122GB, bất khả thi). Data thiếu ví dụ tool-call khi fine-tune sẽ xóa khả năng agent.
- ⚠️ **Overfit khi data ít:** LoRA rank thấp overfit nhanh — 1–3 epoch, theo dõi val loss, thà nhiều data 1 epoch.
- ⚠️ **Thiếu eval / eval sai đường:** không decontaminate = điểm giả; không eval qua đúng LiteLLM+hermes = bỏ sót lỗi production; không có baseline = không biết LoRA giúp hay hại.
- ⚠️ **FP8 vô dụng trên Ampere** (3060 thiếu FP8 tensor core, cần sm_89+). Trên 3060 dùng AWQ/GPTQ/INT8.
- ⚠️ **`MODEL_AUTODETECT` ghi đè:** `detect-gpu.sh` sẽ đưa `MODEL` về Qwen gốc mỗi `start.sh` — đặt `false` khi serve model tự làm.
- ⚠️ **2 container cùng `GPU_MEMORY_UTILIZATION=0.90`** sẽ tranh OOM — trên 3060 gần như không chạy 2 model 7B; server phải hạ util mỗi container hoặc tách `CUDA_VISIBLE_DEVICES`.