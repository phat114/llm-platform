# Tài liệu — Tinh chỉnh & Chiến lược model cho LLM Platform

Cẩm nang kỹ thuật cho việc tùy biến `brain` (Qwen2.5-Coder-7B-AWQ trên vLLM + LiteLLM),
bám sát phần cứng thực tế (RTX 3060 12GB dev / server GPU 24–48GB) và kiến trúc hybrid
`brain` (local) + `brain-pro` (Claude API).

| Tài liệu | Trả lời câu hỏi | Đọc khi |
|---|---|---|
| [fine-tuning-playbook.md](fine-tuning-playbook.md) | "Có những **phương pháp** tinh chỉnh nào, làm được gì, khả thi tới đâu trên phần cứng của tôi?" | Muốn nắm toàn bộ phổ: prompt/decoding → RAG → PEFT (LoRA/QLoRA) → preference (DPO/ORPO) → merge/quant/speculative; kèm bảng VRAM/dữ liệu/công sức và lộ trình. |
| [multitask-strategy.md](multitask-strategy.md) | "Muốn phục vụ **nhiều tác vụ** thì train theo hướng nào / dùng **pattern** nào?" | Cần chọn kiến trúc đa-task: 1 generalist vs N specialist, multi-LoRA serving vs merged model vs RAG/route; kèm khung quyết định + cờ vLLM/LiteLLM cụ thể. |

## Tóm tắt định hướng (một dòng mỗi tài liệu)

- **Tinh chỉnh:** chọn bậc thấp nhất giải quyết được nhu cầu — phần lớn việc (định dạng, giọng
  tiếng Việt, tri thức codebase) làm được **không cần train**; khi train, **QLoRA** là đường ray
  duy nhất vừa 12GB; full-FT/CPT (~112–122GB) là bất khả thi trên phần cứng hiện có.
- **Đa tác vụ:** mặc định **không train** (RAG + tool + persona/route trên 1 base dùng chung);
  khi phải train thì **multi-LoRA** (1 base + N adapter) là trục đúng; đổi base lớn hơn chỉ khi
  chạm trần **suy luận**, không phải khi chạm trần "số task"; đuôi khó đẩy sang `brain-pro`.

> Hai tài liệu này được tổng hợp từ một quá trình khảo sát đa-agent có **phản biện/kiểm chứng
> chéo** các con số kỹ thuật (VRAM, khả năng thật của vLLM multi-LoRA, độ chín công cụ). Các điểm
> đã đính chính so với "kiến thức phổ thông" được đánh dấu trực tiếp trong bài (mục cạm bẫy).
> Trước khi đưa vào production, vẫn cần **verify thực nghiệm** trên đúng version vLLM được pin
> (đặc biệt: LoRA-trên-AWQ, latency multi-LoRA thật, tool-calling hermes của từng adapter).
