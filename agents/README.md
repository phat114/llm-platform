# Agent build — Phase 4

Agent tool-calling gọn, dùng **OpenAI Agents SDK** và trỏ vào gateway của platform
làm "bộ não". SDK lo vòng lặp điều phối tool + retry; ta chỉ khai báo tool bằng
hàm Python — **schema function-calling tự sinh từ type hint + docstring** (không
còn JSON tay). Logic an toàn (khóa trong `AGENT_WORKDIR`) giữ nguyên.

## Tool có sẵn (`tools.py`)
| Tool | Việc |
|------|------|
| `list_dir` | liệt kê file/thư mục |
| `read_file` | đọc file |
| `write_file` | ghi/tạo file |
| `run_shell` | chạy lệnh (build, test, git...) trong WORKDIR |

Mọi thao tác bị **khóa trong `AGENT_WORKDIR`** (chống ghi/đọc bậy ra ngoài).

## Chạy
```bash
cd agents
pip install -r requirements.txt

export LITELLM_MASTER_KEY=sk-...          # trùng .env của platform
export GATEWAY_URL=http://SERVER_IP:4000/v1
export AGENT_WORKDIR=/path/to/du_an       # thư mục agent được phép sửa

python agent.py "Thêm test cho hàm parse_date trong utils.py rồi chạy pytest"
```

## Cờ
- `--model brain` — dùng model **local 3060** (nhanh, miễn phí). Mặc định.
- `--model brain-pro` — route sang **model API ngoài** (task khó, cần điền `ANTHROPIC_API_KEY`).
- `--auto` — chạy shell **không hỏi**. Mặc định agent hỏi `y/N` trước mỗi lệnh shell (an toàn).
- `--max-steps 25` — trần số vòng lặp.

## Mẹo với 7B (local)
Model local 7B hợp task **rõ ràng, phạm vi hẹp**: sửa 1 file, chạy 1 test, generate boilerplate.
Task lớn/mơ hồ → chia nhỏ, hoặc chạy `--model brain-pro`.
