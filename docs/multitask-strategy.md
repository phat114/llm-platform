# Chiến lược đa tác vụ cho "brain" 7B: Train theo hướng nào, dùng pattern nào?

> Bối cảnh cố định: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` serve bằng vLLM (hermes tool-parser, `MAX_MODEL_LEN=32768`, `GPU_MEMORY_UTILIZATION=0.90`, TP=1, prefix caching), gateway LiteLLM đã expose `brain` (GPU local) + `brain-pro` (Claude). Dev RTX 3060 12GB, server deploy 24–48GB. Output tiếng Việt, triết lý self-host gọn nhẹ. Tài liệu này chỉ trả lời: **khi phải phục vụ NHIỀU TÁC VỤ, train/kiến trúc theo hướng nào**.

Câu trả lời một dòng, đọc trước để có la bàn:

> **Mặc định KHÔNG train. Phủ đa task bằng RAG + tool + persona/route trên MỘT base dùng chung. Chỉ train khi eval chứng minh prompt trượt trần. Khi train, mặc định là multi-LoRA (1 base + N adapter, mỗi task 1 adapter) — đây là trục đúng để mở rộng theo số task, đã production-ready trên vLLM. Chỉ đổi base (14B/32B/MoE) khi chạm trần suy luận, không phải khi chạm trần "số task". Đẩy phần đuôi khó/dài sang `brain-pro`. Tránh LoRA-MoE động (X-LoRA/LoraHub/Arrow) trong production — vẫn là research, không chạy native vLLM.**

---

## 1. Khung tư duy: hai trục quyết định + câu hỏi chẩn đoán

Mọi lựa chọn đa task nằm trên **hai trục độc lập**. Lẫn lộn chúng là nguồn gốc quyết định sai.

**Trục A — MỘT generalist vs NHIỀU specialist (mức cô lập trọng số):**
- *Generalist*: một bộ trọng số làm mọi task (prompt-only, hoặc 1 adapter multi-task gộp, hoặc merged model). Ưu: vận hành cực gọn, VRAM không tăng theo task. Nhược: **negative transfer** — task xung khắc kéo nhau xuống, không cô lập chất lượng, thêm task phải đụng lại toàn bộ.
- *Specialist*: mỗi task một adapter riêng (multi-LoRA) hoặc thậm chí model riêng. Ưu: cô lập chất lượng, thêm/sửa task không hại task khác, versioning độc lập. Nhược: cần khâu train + eval + route cho mỗi adapter.

**Trục B — Train (đổi trọng số) vs Không-train (đổi ngữ cảnh/công cụ/định tuyến):**
- *Không-train*: RAG (kiến thức), tool/skill (hành động), persona/system-prompt (hành vi/định dạng), routing (chọn nhánh). Chuyển sự khác biệt task ra khỏi trọng số, đặt vào tầng ứng dụng. Reversible, VRAM gần như không tăng, thêm task tính bằng giờ.
- *Train*: LoRA/QLoRA khi cần **kỹ năng/định dạng cứng** mà prompt không ổn định được. Đắt hơn (data + eval + pipeline), nhưng là cách duy nhất nâng "sàn" tuân thủ format/phong cách của model nhỏ.

**Điểm mấu chốt về "trục chứa" (capacity axes) — để biết đổi cái gì khi bão hoà:**

| Trục chứa | Giới hạn bởi | Dấu hiệu bão hoà | Cách nới |
|---|---|---|---|
| **Kiến thức** (fact, tài liệu) | Params + context | Bịa, sai fact nội bộ | **RAG** (không đổi model) |
| **Kỹ năng/hành vi** (format, phong cách, quy trình) | Fine-tune / LoRA | Format bleed, quên task cũ | **LoRA-per-task** |
| **Chiều sâu suy luận** (agent dài, review sâu) | **Kích thước base** | Lạc bước, không phục hồi lỗi tool-call | Lên **14B/32B** hoặc **brain-pro** |
| **Serving** (nhiều task đồng thời) | VRAM + engine | OOM, xếp hàng, latency tăng | **Multi-LoRA** / thêm GPU |

Hệ quả nền tảng: **một 7B base + N adapter KHÔNG "hết chỗ" theo số task**. Cái bão hoà là *chiều sâu suy luận của base* và *VRAM serving* — không phải "số kỹ năng". Vì vậy chiến lược đa task đúng là **giữ base, nhân adapter**; chỉ đổi base khi chạm trần suy luận.

### Sáu câu hỏi chẩn đoán (trả lời trước khi chọn pattern)

1. **Bao nhiêu task, và chúng LIÊN QUAN hay XUNG KHẮC?** Cùng họ code (sinh/review/tool-calling) → chia sẻ kỹ năng, hợp gộp. Chat VN tự nhiên + JSON extraction cứng + tóm tắt-nén → xung khắc định dạng, cần tách.
2. **Khác nhau ở KIẾN THỨC hay ở KỸ NĂNG/ĐỊNH DẠNG?** Chỉ khác dữ liệu/ngữ cảnh (đều "trả lời dựa trên tài liệu") → RAG + prompt rẻ hơn, khỏi train. Khác *hành vi/format/phong cách* mà prompt không giữ ổn định → mới cần LoRA.
3. **Data mỗi task NHIỀU hay ÍT?** Ít + task liên quan → generalist gộp hưởng positive transfer. Vài task đã dồi dào (chục nghìn mẫu) → specialist thắng generalist.
4. **Danh mục task có ĐỔI THƯỜNG XUYÊN không?** Đổi liên tục → tránh merged (mỗi lần thêm task phải merge + re-quantize + eval lại toàn bộ); ưu tiên multi-LoRA (hot-add) hoặc không-train.
5. **Trần đang chạm là SỐ TASK hay CHIỀU SÂU SUY LUẬN?** Nếu 7B lạc bước ở agent đa bước/review sâu → không adapter nào cứu; phải lên base lớn hoặc route brain-pro.
6. **Ngân sách VRAM/latency?** 3060 12GB: base AWQ ~5.5GB, còn ~4.5–5GB cho KV cache @32k → nút thắt thật là **KV cache**, không phải adapter. Không đủ chỗ cho model 14B+ hay MoE; những cái đó chỉ hợp server ≥24GB.

---

## 2. Cây quyết định chọn pattern

```
BẮT ĐẦU: có một task mới / một danh mục task cần phục vụ
│
├─ Q2: Task khác nhau chỉ ở KIẾN THỨC hoặc CÔNG CỤ,
│      cùng ngôn ngữ & định dạng đầu ra?
│      → CÓ:  RAG multi-collection + tool/skill + persona (KHÔNG train).
│             Đây là MẶC ĐỊNH đầu tiên cho mọi task mới.
│      → KHÔNG (cần đổi format cứng / phong cách / schema ngặt): xuống Q3
│
├─ Q3: Prompt + structured output (guided_json) đã giữ được format ổn định
│      ở tỉ lệ cao chưa? (đo bằng eval, đừng đoán)
│      → RỒI: dừng ở không-train. Xong.
│      → CHƯA (7B trượt format ở đuôi phân phối): cần TRAIN → xuống Q4
│
├─ Q4: Bao nhiêu task cần train, liên quan hay xung khắc, data ra sao?
│      → ÍT task (≈2–4), LIÊN QUAN, ít data/task, danh mục ỔN ĐỊNH:
│           → 1 ADAPTER MULTI-TASK GỘP (generalist). Đơn giản nhất, xoá bài toán route.
│             Nếu muốn latency thấp nhất + 1 artifact: MERGE rồi re-quantize AWQ.
│      → NHIỀU task, có cặp XUNG KHẮC format/phong cách, HOẶC
│        danh mục hay đổi, HOẶC cần versioning/A-B/cách ly chất lượng:
│           → MULTI-LoRA (1 base AWQ + N adapter, mỗi task 1 adapter). ⭐
│
├─ Q5: Có task nào VƯỢT TRẦN SUY LUẬN của 7B?
│      (agent build đa bước, review sâu nhiều file, context rất dài)
│      → CÓ: route sang brain-pro (Claude) — proactive theo classifier,
│             hoặc lên base 14B/32B (chỉ server ≥24GB). LoRA KHÔNG cứu được trần này.
│
└─ Q6 (luôn luôn): dựng DATA-MIX cân bằng + EVAL MATRIX per-task + CI gate
       chống hồi quy chéo. BẮT BUỘC nếu có train, đặc biệt khi có tool-calling.

TRÁNH mọi nhánh: LoRA-MoE động (X-LoRA / LoraHub-runtime / Arrow / PHATGOOSE)
→ research, không native vLLM, overhead ~2×. Nếu cần "trộn kỹ năng":
làm OFFLINE (train multi-task hoặc LoraHub-merge ra 1 adapter tĩnh) rồi serve như adapter thường.
```

---

## 3. Từng pattern: mô tả gọn + khi nào dùng/không

### 3.1 Không-train — RAG multi-collection + tool/skill + persona/routing

**Cơ chế:** chuyển khác biệt task ra khỏi trọng số. Mỗi domain một RAG collection (repo, handbook, chính sách…); mỗi task một bộ tool phù hợp (tận dụng `--enable-auto-tool-choice --tool-call-parser hermes` đã bật); mỗi task một system-prompt/persona; một router mỏng (hoặc chính `brain`) chọn skill + collection + persona. Thêm task = thêm data + tool + 1 nhánh route, reversible, **VRAM base không tăng** khi số task 3 → 30.

**Đính chính con số (fold correction):**
- "VRAM tăng = 0" **chỉ đúng khi embedding/reranker chạy CPU hoặc ngoài GPU**. Nếu chạy embedding/reranker trên **cùng 3060** (cách phổ biến để nhanh): bge-m3/e5-large ~1–2GB, cross-encoder reranker ~1–2GB — trên 3060 vốn đã ~10.8GB dùng cho 7B+KV thì đây là **ràng buộc thật**, có thể phải hạ `gpu-memory-utilization` hoặc đẩy embedding sang CPU/API ngoài. Token prompt tăng còn tăng compute prefill + latency, không chỉ KV cache.
- Latency đến từ **số vòng LLM**, không từ VRAM: persona tĩnh +~0; router phân loại 1 bước +0.3–0.8s; agentic multi-hop 2–6 vòng +2–8s (3060). Muốn nhanh phải giảm hop.

**Cạm bẫy fallback (fold correction quan trọng):** `fallbacks: [{"brain":["brain-pro"]}]` của LiteLLM **chỉ kích hoạt khi brain LỖI** (exception/timeout/OOM/5xx/rate-limit). Nó **KHÔNG tự phát hiện "task khó vượt 7B"** để chuyển. Muốn route theo *độ khó*, phải có **logic tường minh**: classifier/planner tự quyết định gọi `brain-pro` (chọn model ở tầng ứng dụng). Ngoài ra trong repo hiện tại, dòng `fallbacks` đang **bị comment** và `brain-pro` (claude-sonnet-5) **chưa có ANTHROPIC_API_KEY** — nên fallback mới ở dạng mẫu chưa kích hoạt.

| | Không-train (RAG/tool/persona/route) |
|---|---|
| Ưu | VRAM thêm ≈ 0 khi tăng task; thêm task tính bằng giờ–ngày; reversible & dễ A/B; không catastrophic forgetting; dùng lại nguyên hạ tầng hermes + prefix caching. |
| Nhược | Bị chặn trần bởi năng lực nền 7B — RAG bơm *kiến thức* nhưng không bơm *kỹ năng suy luận / tuân thủ format cứng*; latency tăng theo hop; agentic động phi tất định, khó test, dễ loop; chọn tool sai khi >10–15 tool chồng lấn. |
| **NÊN** | Task khác nhau chủ yếu ở **kiến thức/công cụ**, cùng ngôn ngữ & định dạng (Q&A tài liệu, tra chính sách, code trên repo). Danh mục đổi thường xuyên. **Lựa chọn mặc định đầu tiên cho mọi task mới.** |
| **KHÔNG** | Cần schema JSON ngặt ổn định ở tỉ lệ cao; cần suy luận vượt 7B; routing chính xác giữa >10–15 skill; cần latency thấp tất định (tránh agentic động). |

### 3.2 Generalist — một adapter multi-task gộp (mixed SFT)

**Cơ chế:** gộp N task vào **một dataset trộn**, train **một adapter** trên base bf16 (KHÔNG train trên AWQ — AWQ chỉ để serve). Task phân biệt qua system prompt. Ăn **positive transfer** khi task liên quan + ít data; nhưng chịu **negative transfer** khi task xung khắc (chat VN dài ↔ JSON cụt; tóm tắt-nén ↔ QA bám chi tiết) và **catastrophic forgetting** năng lực gốc nếu thiếu replay.

Núm vặn quyết định: mixing theo nhiệt độ `p_i ∝ n_i^(1/T)`, thực chiến **T≈1.5–2**, cap task áp đảo ≤30–35%, sàn ~300–500 mẫu/task; **replay 5–15%** instruction/code gốc + giữ style tool-call hermes (bắt buộc, không thì tool-calling thoái hoá); loss chỉ tính phần assistant, mask tool-result.

**Đính chính (fold correction) — về "bão hoà 3–6 task":** con số "3–6 task rồi negative transfer" là **kinh nghiệm định tính, không phải mốc cứng** — ngưỡng phụ thuộc độ tương đồng task, dung lượng model, chất lượng data. Về chất, luận điểm đúng: task càng khác format/ngôn ngữ đích, trần càng thấp.

| | Generalist gộp (1 adapter) |
|---|---|
| Ưu | Ops gọn: 1 adapter, 1 version, VRAM ≈ base. Transfer dương giúp task ít data. Nếu merge → 0% overhead, route không đổi. |
| Nhược | Interference kéo task xung khắc xuống, không cô lập; thêm/sửa 1 task phải **train lại toàn bộ**; khó debug task nào hỏng. |
| **NÊN** | ≈2–4 task **liên quan** + ít data/task, danh mục ổn định, ưu tiên đơn giản vận hành. |
| **KHÔNG** | Có cặp task xung khắc format/giọng rõ rệt và per-task eval tụt dù đã chỉnh mixing/replay; vài task đã dồi dào data; cần versioning/rollback/A-B độc lập. |

### 3.3 Merged single-model (task arithmetic / TIES / DARE-TIES)

**Cơ chế:** làm việc trên delta weights `τ_i = θ_finetune_i − θ_base`. TIES trim top-k% + elect-sign + merge phần đồng dấu; DARE-TIES thêm drop ngẫu nhiên p% + rescale (p≈0.5–0.9) — recipe mạnh nhất hiện nay. Ràng buộc cứng: **mọi fine-tune phải cùng base checkpoint + cùng tokenizer**.

**Bắt buộc với stack AWQ:** không merge trực tiếp trên AWQ. Quy trình: base fp16 → merge (mergekit, chạy được cả chế độ CPU/low-VRAM trên 3060, ~10–30 phút) → **re-quantize AWQ** (cần calibration set, ~20–60 phút trên 3060) → **eval hồi quy** → serve.

**Đính chính latency (fold correction):** merged = latency 1 model đơn (đúng). Nhưng tốc độ decode 1 luồng 7B AWQ trên 3060 **~30–55 tok/s** (đỉnh ~80 chỉ là ceiling lý thuyết ~360GB/s bw, hiếm khi đạt vì dequant AWQ tốn thêm) — **không phải 40–80**.

| | Merged single-model |
|---|---|
| Ưu | 1 artifact, latency thấp nhất, VRAM không tăng theo task, route không đổi (chỉ `brain`). |
| Nhược | Interference vĩnh viễn (không núm tắt task); thêm task phải merge + re-quant + eval lại toàn bộ; **sign-election dễ phá format tool-call hermes** — phải eval end-to-end tool-call, chat/perplexity không bắt được lỗi này; DARE-TIES nhạy hyperparam (dò p, λ). |
| **NÊN** | ≤3–4 task **cùng họ** (code + review + agent), muốn 1 artifact + latency thấp nhất, danh mục ổn định. |
| **KHÔNG** | Task xa nhau; danh mục biến động; cần cách ly/versioning/A-B; cần tool-calling tin cậy cao (rủi ro hồi quy format). |

### 3.4 Multi-LoRA serving trên vLLM ⭐ (xương sống đa task khi phải train)

**Cơ chế:** base AWQ nạp **một lần** (frozen); mỗi task một adapter LoRA nhỏ. Request mang `model=<tên_adapter>`, vLLM cắm đúng adapter. **Kernel Punica/SGMV (kỹ thuật S-LoRA)** gom **nhiều adapter khác nhau trong CÙNG một batch/forward** — base chạy dày một lần, phần LoRA gather-scatter theo từng sequence. Đây là lý do 1 GPU nhỏ phục vụ N task với chi phí gần bằng 1 model. Prefix caching vẫn hoạt động cho system prompt/tài liệu dùng chung.

**Đính chính then chốt (fold corrections) — khả năng THẬT của vLLM multi-LoRA:**
- `--max-loras` là **TRẦN số adapter distinct trong 1 batch step**, KHÔNG phải "công tắc giảm throughput". Đặt `max-loras=4` mà thực tế mỗi batch chỉ gặp 1 adapter thì **gần như không mất throughput**. Tổn thất thật đến từ **độ đa dạng adapter thực tế trong batch** (nhiều adapter khác nhau → mỗi adapter được batch nhỏ hơn → SGMV kém hiệu quả), không phải từ giá trị trần. **Con số "10–30%" là bịa, không có cơ sở đo**; tài liệu vLLM/Punica mô tả overhead quản lý adapter là "minimal", giữ ~80% throughput ngay cả khi phục vụ >1000 LoRA (có nén). **Đặt `max-loras ≈ số adapter distinct kỳ vọng đồng thời**, không phải mặc định sợ "cao".
- Adapter KHÔNG phải nút thắt VRAM. Số adapter **đăng ký** điều khiển bởi `--max-cpu-loras` (pool RAM CPU, hàng chục–hàng trăm, swap vào GPU vài ms khi cần); số adapter **active/batch** là `--max-loras`. Nút thắt thật trên 3060 là **KV cache @32k**.
- Kích thước adapter phụ thuộc target_modules: chỉ attention (q,v) rank16≈17MB; q,k,v,o rank64≈130MB; **LoRA-all-linear (q,k,v,o,gate,up,down) — cấu hình hay dùng để fine-tune code cho chất — rank16≈80MB, rank64≈320MB, tức GẤP ~2 lần các khoảng "40–120MB" thường trích**. Vẫn nhỏ so với KV cache.

**Cảnh báo AWQ + LoRA:** vLLM **có** hỗ trợ `--enable-lora` trên base AWQ (qua awq_marlin, Ampere sm_86 OK). Nhưng bạn **train QLoRA trên base bf16 rồi serve adapter trên base AWQ 4-bit** → base lúc train ≠ base lúc serve → **rủi ro lệch chất lượng nhẹ**, nhất là task đòi chính xác (JSON extraction, review). Nhánh AWQ+LoRA **ít battle-tested hơn** LoRA-trên-fp16 và **kén version vLLM**. **Bắt buộc eval từng adapter trên chính base AWQ** trước khi tin dùng; nếu tụt đáng kể → serve task đó trên base fp16 (chỉ vừa server ≥24GB) hoặc đẩy brain-pro.

**Giới hạn tinh tế:** `--tool-call-parser hermes` là cấu hình **cấp server**, không đổi theo adapter. May mắn mọi adapter Qwen2.5 dùng hermes → không xung đột. Nhưng mọi task tool-calling phải theo cùng convention hermes; muốn format khác → phải chạy instance vLLM thứ hai.

| | Multi-LoRA serving |
|---|---|
| Ưu | Base×1 + adapter vài chục–vài trăm MB; **cách ly chất lượng cao**; thêm task = 1 adapter + 1 route, hot-add không restart; batch trộn nhiều task (S-LoRA); versioning/A-B/rollback theo adapter. |
| Nhược | Overhead nhẹ khi batch trộn nhiều adapter distinct; AWQ+LoRA cần verify; router phức tạp hơn merged; mỗi adapter vẫn dùng chung base đóng băng nên chung trần suy luận 7B. |
| **NÊN** | Nhiều task khác **hành vi/format** cần chất lượng ổn định; cần cách ly + versioning; phần cứng eo hẹp (3060) phải phục vụ ≥3–4 task — gần như **lựa chọn duy nhất** vừa VRAM; thêm/bớt task liên tục. |
| **KHÔNG** | Task chỉ khác ngữ cảnh (RAG rẻ hơn); chỉ 1–2 task prompt đã đủ; task quá khó vượt 7B (route brain-pro); cần format tool-call khác nhau (phải instance riêng). |

### 3.5 Đổi base khi chạm trần suy luận: 14B / 32B / MoE

Đúng khi trần là **chiều sâu suy luận** (agent build đa bước, review sâu), KHÔNG phải số task.

| Model | Weights AWQ | VRAM thực (kèm KV) | 3060 12GB | Server 24GB | Server 48GB |
|---|---|---|---|---|---|
| 7B AWQ | ~5–5.5GB | ~8–11GB @32k | ✅ mặc định | thừa (dồn adapter) | thừa |
| 14B AWQ | ~9–10GB | ~13–16GB @16k | ❌ | ✅ @16k | ✅ @32k |
| 32B AWQ | ~18–20GB | ~22–26GB @16k | ❌ | ⚠️ sát trần @8–16k | ✅ |
| Qwen3-30B-A3B (MoE) | ~16–18GB | active ~3B → decode nhanh | ❌ (weights > VRAM) | ⚠️ chật, ctx ngắn | ✅ throughput cao |

MoE: active params thấp → decode nhanh + throughput cao khi phục vụ nhiều task đồng thời, **nhưng VRAM = TỔNG params** (không giải bài toán 12GB) và **LoRA-trên-MoE còn ít công cụ chín (research)** — lợi ích chỉ hiện ở server ≥40–48GB, và nên serve base + không LoRA-trên-MoE cho tới khi công cụ chín. Base lớn hơn cũng juggle prompt-only đa task tốt hơn và vẫn gắn multi-LoRA được (miễn còn VRAM cho KV).

### 3.6 Định tuyến adapter: 3 tầng (chọn 1 adapter → router học → LoRA-MoE)

- **(a) Routing tường minh ở LiteLLM** — client khai task qua tên model (`brain-code`, `brain-review`…). 0 latency thêm, 0 route sai, debug hiển nhiên. **Mặc định cho ~90% việc nội bộ** (bạn viết cả client lẫn server).
- **(b) Router học** — classifier nhỏ (embedding multilingual-e5-small ~118M + logistic/kNN, hoặc LLM-as-router dùng chính brain) đoán intent khi có **một ô chat tự do**. Đặt micro-service FastAPI **trước** LiteLLM; threshold thấp → default an toàn (`brain-vi-chat`) hoặc escalate. LiteLLM **không** làm phân loại intent hộ.
- **(c) LoRA-MoE động** — X-LoRA (gating per-token/layer, ~2× compute), LoraHub (gradient-free, thực chất merge OFFLINE ra adapter tĩnh), Arrow, PHATGOOSE. **Fold correction: đều là RESEARCH, KHÔNG native vLLM, KHÔNG production.** Nếu cần "trộn kỹ năng": làm offline (train multi-task hoặc LoraHub-merge ra 1 adapter tĩnh) rồi serve như adapter thường — giữ được sự đơn giản của (a).

### 3.7 Data-mix + eval matrix chống hồi quy chéo (BẮT BUỘC nếu train)

Không phải kỹ thuật train mới mà là **lớp kỷ luật bao quanh** mọi lựa chọn train ở trên. Với 7B (capacity chật) + tool-calling hermes trong danh mục, **bỏ qua pattern này gần như chắc chắn dẫn tới regression âm thầm**.

- **Data-mix:** gắn `task_id` mọi mẫu; temperature sampling T≈1.5–2 cap task lớn ≤30–35%; upsample task hiếm (≤5×); **replay 10%** general/code + giữ style hermes; tool-calling là task hạng nhất ≥10–15% (dễ rớt nhất).
- **Eval matrix + CI gate:** mỗi task held-out 50–200 mẫu sạch (không rò rỉ); ma trận task×metric có cột **delta** vs baseline; **cổng promote cứng**: không task nào rớt quá ngưỡng (vd −2%), **tool-call valid-JSON rate ≥95% (hard gate)**. Blue-green promote (đổi alias/lora hot-swap), rollback tức thì.

**Đính chính đo lường (fold corrections):**
- **Noise band phải lấy từ SEED HUẤN LUYỆN khác nhau (≥3 seed train), không phải seed decoding.** Với `temperature=0` (greedy) decoding gần như tất định, chạy lại cùng model cho kết quả gần y hệt — 2 mẫu quá ít để ước lượng độ lệch. Dùng ≥3 seed train hoặc khoảng tin cậy nhị thức/bootstrap trên eval set.
- **Đừng eval single-stream.** vLLM dùng continuous batching xử lý song song hàng trăm prompt → throughput tổng hàng trăm–hơn nghìn tok/s. Một suite ~700 gen thường **~5–15 phút NGAY TRÊN 3060** (không phải 20–50 phút), vài phút trên server. Dùng offline batched generation với `max_num_seqs` cao.
- **LLM-judge dùng `brain-pro`** để không tranh GPU + khách quan hơn; pairwise + swap vị trí giảm bias; đừng để model đang eval tự chấm task subjective.

---

## 4. Bảng so sánh tổng các pattern

| Pattern | Số task hợp | 1 model / N | VRAM 3060 12GB | Vận hành | Chất lượng/task | Độ chín | Khi nào chọn |
|---|---|---|---|---|---|---|---|
| **Không-train (RAG/tool/persona)** | rất nhiều | 1 base | base + (embed/rerank ~1–4GB nếu để GPU) | Thấp | TB (trần 7B) | Chín (agentic động cận-production) | **Mặc định** khi task khác ở kiến thức/công cụ, cùng format |
| **Generalist gộp (1 adapter)** | 2–4 liên quan | 1 model | ≈ base (~5.5GB + KV) | Thấp (route không đổi) | TB (thỏa hiệp) | Chín (T5/FLAN kinh điển) | Ít task liên quan, ít data, ổn định |
| **Merged single-model** | ≤3–4 cùng họ | 1 model | ≈ base | Thấp serve / Vừa–Cao pipeline | TB (interference, DARE-TIES giảm không xoá) | Chín (mergekit); nhạy hyperparam | 1 artifact + latency thấp nhất, danh mục ổn định |
| **Multi-LoRA ⭐** | nhiều | 1 base + N adapter | base + adapter (vài chục–vài trăm MB), **KV là nút thắt** | Vừa (router theo tên) | **Cao & cách ly** | **Production-ready** (AWQ+LoRA cần verify) | Nhiều task khác format, cần cách ly/versioning, GPU nhỏ ≥3–4 task |
| **Lên 14B/32B AWQ** | nhiều | 1 base lớn | ❌ không vừa | Thấp (đổi .env) | Cao (reasoning tốt hơn) | Chín | Trần là **suy luận**, server ≥24GB |
| **MoE (Qwen3-30B-A3B)** | nhiều đồng thời | 1 base MoE | ❌ không vừa | Vừa | Cao throughput | Serve chín; **LoRA-MoE research** | Server ≥48GB, cần throughput nhiều task |
| **Multi-model distill nhỏ** | 1–2 nóng | N model | ⚠️ chỉ 1 model nhỏ cạnh brain | Cao | Cao (cô lập) | Chín | 1–2 task cực nóng/đơn giản cần cực nhanh |
| **brain-pro (Claude)** | đuôi khó/dài | API | 0 VRAM | Thấp (có sẵn) | Cao nhất | Chín | Agent phức tạp, context rất dài, task hiếm |
| **LoRA-MoE động (X-LoRA/LoraHub-rt/Arrow)** | — | — | rủi ro OOM/2× | Cao | không chắc | **Research — tránh** | Chỉ R&D; production thì làm offline ra adapter tĩnh |

---

## 5. So sánh trực diện cho GPU NHỎ (3060 12GB): 4 lựa chọn cốt lõi

| Tiêu chí | Multi-LoRA serving | Merged single-model | Generalist mixed-SFT | RAG/route không-train |
|---|---|---|---|---|
| **VRAM đa task** | base + vài trăm MB, KV là nút thắt | đúng 1 model | ≈ 1 model | base + (embed/rerank nếu để GPU) |
| **Thêm task mới** | 1 adapter + 1 route, hot-add | **merge + re-quant + eval lại toàn bộ** | **train lại toàn bộ** | thêm data/tool/prompt (giờ) |
| **Cách ly chất lượng** | **Cao** | Thấp (interference) | Thấp | Cao (nhưng trần 7B) |
| **Latency** | overhead nhẹ khi batch trộn nhiều adapter | **1 model, thấp nhất** (~30–55 tok/s AWQ 3060) | 1 model | tăng theo số hop route |
| **Tool-calling** | mỗi adapter train giữ hermes; parser dùng chung | **dễ vỡ, phải eval e2e** | replay hermes bắt buộc | dùng nguyên hermes, không train |
| **Versioning/A-B/rollback** | dễ, theo adapter | thô, cả model | thô, cả adapter | dễ, theo config |
| **Công sức chính** | train + eval N adapter; verify AWQ+LoRA | dò p/λ + re-quant + eval hồi quy | curate + cân bằng data + eval per-task | dựng RAG + skill registry |
| **Rủi ro ẩn** | AWQ+LoRA kén version | sign-election phá format | negative transfer âm thầm | embed/rerank ăn VRAM; agentic loop |

**Kết luận cho 3060:** với "N task khác format", **multi-LoRA là trục đúng** — VRAM adapter không đáng kể, cách ly chất lượng, thêm task rẻ. Merged chỉ thắng khi ≤3–4 task cùng họ + cần latency thấp nhất + danh mục ổn định. Generalist gộp là bước đệm đơn giản khi task liên quan. RAG/không-train phủ trước phần lớn task "khác kiến thức" mà không tốn ngày train nào.

---

## 6. KHUYẾN NGHỊ DỨT KHOÁT cho setup của bạn

**Pattern trục:** **Multi-LoRA trên 1 base Qwen2.5-Coder-7B-AWQ dùng chung, bọc bởi lớp không-train (RAG/tool/persona) làm mặc định, và `brain-pro` làm van xả cho đuôi khó.** Không nhồi nhiều task vào một trọng số; không đụng LoRA-MoE động; không đổi base khi chỉ thiếu "số task".

### Lộ trình 4 bước

**Bước 0 — Dựng lưới an toàn (làm trước cả khi train adapter đầu tiên).** Data-mix có `task_id`; eval matrix per-task (50–100 mẫu/task trên 3060) với **tool-call hard gate ≥95%**; noise band từ ≥3 seed train; chạy eval bằng **batched offline** (~5–15 phút/suite trên 3060). Đây là chi phí một lần, sau đó mỗi vòng train được "bọc lưới".

**Bước 1 — Phủ tối đa bằng KHÔNG-train.** Tách `SYSTEM`/`tools.TOOLS` trong `agents/agent.py` thành **SKILLS registry** (persona + tool + collection mỗi skill). Dựng RAG multi-collection (embedding chạy **CPU hoặc process TEI riêng** để không cắn KV cache trên 3060). Router phân loại rẻ = chính `brain`. Ép JSON bằng `guided_json`/`response_format` cho extraction/classification. Hạ `max_turns` trong agent từ 25 xuống **8–12** cho task thường để chặn loop. **Chỉ những task eval chứng minh prompt trượt trần mới đi tiếp Bước 2.**

**Bước 2 — Train multi-LoRA cho task ROI cao mà prompt "trượt".** Ưu tiên `extract` (JSON schema), `review` (bới lỗi), `vi-chat` (giọng tự nhiên). Train QLoRA trên **base bf16** (trên cloud/server 24–48GB để có seqlen dài, r=32 alpha=64, gradient checkpointing), bê adapter về serve trên 3060. **Verify từng adapter trên base AWQ** so với fp16 trước khi promote — nếu lệch đáng kể, giữ task đó ở prompt-only/brain-pro. Nếu ≤3–4 task liên quan và muốn tối giản, cân nhắc **1 adapter gộp**; chỉ tách adapter khi per-task eval cho thấy interference âm.

**Bước 3 — Route đuôi khó sang brain-pro + mở rộng theo phần cứng.** Trên 3060 giữ 7B AWQ. Lên server: nếu trần là suy luận → base 14B (24GB)/32B (48GB) vẫn gắn multi-LoRA; nếu cần throughput nhiều task đồng thời và ≥48GB → cân nhắc MoE (serve base, chưa LoRA-trên-MoE).

### Phân bổ task giữa `brain` local và `brain-pro`

Nguyên tắc: **local ăn thân phân phối (nhẹ, tần suất cao, nhạy cảm dữ liệu); brain-pro ăn đuôi (khó/dài/hiếm).**

| Task | Đi đâu | Lý do |
|---|---|---|
| Chat VN, tóm tắt ngắn | brain / brain-vi (local) | 7B thừa sức, volume cao |
| Phân loại / trích xuất schema | brain-extract (LoRA) + guided_json | LoRA ép format ổn định |
| Sinh code đoạn ngắn/vừa | brain-code (LoRA) | Coder-7B mạnh sẵn |
| Review code sâu, nhiều file | **brain-pro** | Vượt trần suy luận 7B |
| Agent build đa bước phức tạp | **brain-pro**; agent đơn giản → local | 7B lạc bước khi chuỗi tool dài |
| QA tài liệu nội bộ (RAG) | brain + RAG; context rất dài → brain-pro | Không cần train; đuôi dài đẩy pro |

**Cơ chế route "khó → pro" phải TƯỜNG MINH:** classifier/planner chấm độ khó/độ dài input rồi **chọn model ở tầng ứng dụng** (proactive). `fallbacks` của LiteLLM chỉ là van **reactive khi lỗi** (OOM/timeout), không phát hiện độ khó — và hiện đang comment, cần điền `ANTHROPIC_API_KEY` + bỏ comment mới chạy.

### Land vào vLLM + LiteLLM (cờ cụ thể)

**vLLM** (qua `VLLM_EXTRA_ARGS` trong `scripts/vllm-entrypoint.sh`, giữ nguyên hermes/max_model_len/gpu_util/prefix caching):
```
--enable-lora \
--max-loras 3 \           # 3060: đặt ≈ số adapter distinct kỳ vọng đồng thời (2–4), KHÔNG phải "càng thấp càng an toàn"
--max-lora-rank 32 \      # = rank lớn nhất bạn train
--max-cpu-loras 16 \      # pool RAM, đăng ký hàng chục adapter thoải mái
--lora-modules code=/adapters/code review=/adapters/review \
               vi-chat=/adapters/vi-chat extract=/adapters/extract
# hot-add không restart: ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 + POST /v1/load_lora_adapter
```
Mount `- ./adapters:/adapters:ro` vào container vLLM. Giữ `--served-model-name brain`.

**Nếu chọn merged (thay vì multi-LoRA):** sau merge+re-quant AWQ, **GIỮ NGUYÊN `SERVED_MODEL_NAME=brain`** và trỏ **biến `MODEL`** trong `.env` (hiện `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`) sang weights merged mới. `MODEL` chọn trọng số; `SERVED_MODEL_NAME` là alias công khai phải giữ `brain` để `litellm.yaml` và mọi client không phải đổi. (Đừng đổi `SERVED_MODEL_NAME` — nó phải khớp `openai/brain` mà LiteLLM trỏ tới.)

**LiteLLM** (`config/litellm.yaml`) — mỗi adapter = một `model_name`:
```yaml
model_list:
  - model_name: brain          # base, chat chung
    litellm_params: {model: openai/brain, api_base: http://vllm:8000/v1, api_key: os.environ/VLLM_API_KEY}
  - model_name: brain-code
    litellm_params: {model: openai/code, api_base: http://vllm:8000/v1, api_key: os.environ/VLLM_API_KEY}
  - model_name: brain-review
    litellm_params: {model: openai/review, api_base: http://vllm:8000/v1, api_key: os.environ/VLLM_API_KEY}
  # extract, vi-chat tương tự
  - model_name: brain-pro       # Claude — cần điền ANTHROPIC_API_KEY
    litellm_params: {model: anthropic/claude-..., api_key: os.environ/ANTHROPIC_API_KEY}
# reactive fallback (bỏ comment sau khi có key): chỉ kích hoạt khi brain LỖI, KHÔNG theo độ khó
# litellm_settings:
#   fallbacks: [{"brain": ["brain-pro"]}]
```

---

## 7. Cạm bẫy đa tác vụ (checklist)

1. **Negative transfer âm thầm** — thêm task B làm rớt task A, loss tổng đẹp che 1 task tụt. Chống: **eval per-task + cột delta**, không nhìn loss tổng. Noise band từ ≥3 seed train, không từ decoding.
2. **Quên tool-calling** — thêm data text-only (chat/summarize) làm JSON tool-call rate rớt (vd 97%→88%). Chống: tool-calling là task hạng nhất ≥10–15% mix + replay hermes + **hard gate ≥95% parse e2e** (parser server không cứu được nội dung model sinh sai tag).
3. **Merge phá format** — TIES/DARE sign-election phá tag tool-call; chat/perplexity KHÔNG bắt được. Chống: eval end-to-end tool-call trước/sau merge.
4. **AWQ + `--enable-lora`** — train bf16 / serve AWQ 4-bit → lệch chất lượng; nhánh này kén version vLLM, ít battle-tested. Chống: **eval từng adapter trên chính base AWQ**; pin version; task nhạy → fp16 (server) hoặc brain-pro.
5. **Ngộ nhận VRAM adapter** — nút thắt trên 3060 là **KV cache @32k**, KHÔNG phải adapter (adapter LoRA-all-linear r32 chỉ ~160MB; r64 ~320MB). `--max-loras` cao không tự giảm throughput; tổn thất đến từ **độ đa dạng adapter thực tế trong batch**. Đặt `max-loras ≈ số adapter distinct đồng thời`.
6. **RAG ăn VRAM ngầm** — embedding/reranker để chung 3060 tốn 1–4GB, có thể ép phải hạ `gpu-memory-utilization`. Chống: chạy embed/rerank trên CPU hoặc process/API riêng. Token prompt dài tăng cả prefill compute lẫn KV.
7. **Ngộ nhận fallback** — `fallbacks` LiteLLM chỉ reactive khi lỗi, không phát hiện độ khó; và hiện đang comment/thiếu key. Route theo độ khó phải viết logic tường minh.
8. **Agentic động phi tất định** — dễ loop, khó test hồi quy, latency khó đoán. Chống: chốt router phân loại tĩnh + skill tường minh trước; `max_turns` 8–12.
9. **Đổi base nhầm trục** — lên 14B/32B khi chỉ thiếu "số task" (đáng lẽ thêm adapter) là lãng phí; ngược lại nhồi thêm adapter khi thật sự chạm trần **suy luận** thì vô ích. Phân biệt: task nào cũng "tàm tạm không xuất sắc" → tách adapter; agent không phục hồi lỗi tool → lên base/brain-pro.
10. **LoRA-MoE là bẫy công sức** — X-LoRA/LoraHub-runtime/Arrow/PHATGOOSE không native vLLM, mất prefix caching + kernel batching, overhead ~2×, vẫn research. Cần trộn kỹ năng thì làm **offline** ra 1 adapter tĩnh.