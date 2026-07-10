# Định dạng dữ liệu train (JSONL)

Mỗi dòng là **một mẫu hội thoại** đã có sẵn câu trả lời "vàng":

```json
{"task_id": "extract", "messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

## Trường
| Trường | Bắt buộc | Ý nghĩa |
|--------|----------|---------|
| `task_id` | ✅ | Tên task (khớp skill: `build`/`review`/`chat`/`extract`...). Dùng để **cân bằng data-mix** và **eval per-task**. |
| `messages` | ✅ | Danh sách turn theo chuẩn chat. Có thể nhiều turn; loss chỉ tính phần `assistant` (xem `train_on_responses_only`). |

## Quy tắc quan trọng
- **Giữ đúng chat template Qwen2.5 (ChatML).** KHÔNG tự viết tay chuỗi `<|im_start|>`; `train_qlora.py`
  gọi `tokenizer.apply_chat_template(...)` để sinh. Ở đây chỉ cần cung cấp `messages` sạch.
- **Tool-calling (task `build`/agent):** turn `assistant` chứa đúng khối hermes mà parser server bóc:
  ```
  <tool_call>
  {"name": "read_file", "arguments": {"path": "config.py"}}
  </tool_call>
  ```
  Đây chính là format `--tool-call-parser hermes` đang dùng. Giữ nguyên `<tool_call>...</tool_call>`
  và JSON args hợp lệ — nếu sai, adapter sẽ làm hỏng tool-calling.
- **Ngôn ngữ:** câu trả lời `assistant` bằng tiếng Việt (đúng mục tiêu output của platform).
- **Decontamination:** loại mọi mẫu trùng (≥13-gram) với eval/benchmark set — nếu không, điểm eval là giả.
- **Chất lượng > số lượng:** 300–1.500 mẫu/task sạch đã đổi được style/format rõ; nguồn tốt nhất là
  distill từ `brain-pro` (Claude) + log thật đã lọc.

## ⚠️ Giới hạn tool-calling & multi-turn (đọc trước khi train adapter `build`/agent)
- **Khớp train ↔ serve:** khi serve, agent gửi kèm định nghĩa tool → chat template Qwen2.5 chèn
  block `# Tools` vào system. Nếu mẫu train KHÔNG có phần định nghĩa tool đó, phân phối train/serve
  lệch → tool-calling tụt. Cách đúng: nhúng đúng định nghĩa tool vào `system` của mẫu train (hoặc
  dùng `messages` có `assistant.tool_calls` structured + truyền `tools=[...]` khi sinh). Mẫu
  `build` trong `example.jsonl` hiện là bản TỐI GIẢN (chỉ minh hoạ format hermes), chưa nhúng tool-def.
- **Masking multi-turn:** `train_qlora.py` mask theo một `instruction_part` — chỉ đúng cho
  **single-turn** (1 user → 1 assistant). Hội thoại nhiều lượt có role `tool` cần mask thêm vùng
  tool-result (truyền `instruction_part` dạng list mọi header không-phải-assistant), nếu không
  model sẽ bị tính loss trên cả kết quả tool. Giữ mẫu tool-calling **single-turn** cho tới khi cần.

## Nguồn sinh nhanh
- Gọi `brain-pro` qua gateway (`model="brain-pro"`) để sinh `assistant` chất lượng cao.
- Xuất log production qua LiteLLM `success_callback` → JSONL, rồi gắn `task_id` + lọc mẫu xấu.
