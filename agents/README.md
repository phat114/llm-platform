# Agent đa-kỹ-năng — Phase 4

Agent tool-calling gọn trên **OpenAI Agents SDK**, trỏ vào gateway của platform làm "bộ não".
Từ Bước 1 (chiến lược đa tác vụ — xem [docs/multitask-strategy.md](../docs/multitask-strategy.md)),
agent được tổ chức thành **SKILLS registry**: mỗi tác vụ là một *skill* gồm persona + bộ tool +
model route. Một **router** (dùng chính `brain`) chọn skill phù hợp cho mỗi yêu cầu.

Thêm một tác vụ mới = thêm **một Skill** trong `skills.py` — **KHÔNG train lại gì**. Đây là cách
"phủ đa task không-train" trước khi phải đụng tới LoRA.

```
task ──► router.choose_skill (brain, structured output)
            │  chọn skill theo mô tả
            ▼
        Skill { persona + tools + model + max_turns }
            │  dựng Agent (OpenAI Agents SDK)
            ▼
        Runner.run_sync ──► tool-calling loop ──► kết quả
```

## Skills có sẵn (`skills.py`)
| Skill | Tool | Model route | Dùng khi |
|-------|------|-------------|----------|
| `build` | read/write/list/shell | `brain` (local) | Động tới file / chạy lệnh (mặc định cho lập trình) |
| `review` | read_file, list_dir (chỉ đọc) | `brain-pro` (Claude) | Review code sâu — cần `ANTHROPIC_API_KEY` |
| `chat` | (không) | `brain` | Hỏi đáp/giải thích tiếng Việt; **fallback mặc định** |
| `extract` | (không) | `brain` | Trích xuất/phân loại ra JSON |

Xem chi tiết: `python agent.py --list-skills`.

> **RAG collection** (`rag.py`): skill có `collection` (vd `build`/`review` = `codebase`) được gắn
> thêm tool `search_codebase` để **truy hồi** code liên quan thay vì mò `list_dir`/`read_file`.
> Embedding chạy **CPU** (FastEmbed) + vector store **Chroma** — không tốn VRAM, không thêm container.
> Cần cài deps (`pip install fastembed chromadb`) và **index một lần**:
> ```bash
> export AGENT_WORKDIR=/path/to/du_an
> python rag.py index          # quét WORKDIR → chunk → embed → Chroma (.rag/)
> python rag.py search "..."    # thử truy hồi
> ```
> Chưa index thì tool báo rõ và agent vẫn chạy bình thường với các tool file.

## Tool có sẵn (`tools.py`)
| Tool | Việc |
|------|------|
| `list_dir` | liệt kê file/thư mục |
| `read_file` | đọc file |
| `write_file` | ghi/tạo file |
| `run_shell` | chạy lệnh (build, test, git...) trong WORKDIR |

Schema function-calling **tự sinh từ type hint + docstring** (không khai báo JSON tay). Mọi thao
tác bị **khóa trong `AGENT_WORKDIR`** (`_safe_path` chống đọc/ghi ra ngoài).

## Router (`router.py`)
"Router phân loại rẻ = chính `brain`": một lần gọi model với **structured output**
(`response_format` json_schema, ép chọn đúng một tên skill hợp lệ). Fallback nhiều tầng: nếu
endpoint không hỗ trợ json_schema → thử `json_object`; nếu vẫn lỗi/không hợp lệ → `DEFAULT_SKILL`
(`chat`, an toàn, không tool). Router **không bao giờ làm hỏng CLI** khi lỗi mạng.

## Chạy
```bash
cd agents
pip install -r requirements.txt

export LITELLM_MASTER_KEY=sk-...          # trùng .env của platform
export GATEWAY_URL=http://SERVER_IP:4000/v1
export AGENT_WORKDIR=/path/to/du_an       # thư mục agent được phép thao tác

# Router tự chọn skill:
python agent.py "Thêm test cho parse_date trong utils.py rồi chạy pytest"
# Ép skill (bỏ qua router):
python agent.py --skill review "Xem app.py có lỗi bảo mật gì không"
python agent.py --list-skills
```

## Cờ
- `--skill NAME` — ép skill (`build`/`review`/`chat`/`extract`), bỏ qua router.
- `--model NAME` — đè model của skill (`brain` local, `brain-pro` API ngoài).
- `--max-steps N` — đè `max_turns` của skill.
- `--auto` — chạy shell **không hỏi**. Mặc định hỏi `y/N` trước mỗi lệnh shell (an toàn).
- `--list-skills` — in registry rồi thoát.

## Thêm một skill mới
Sửa `skills.py`, thêm một mục vào `SKILLS`:
```python
"translate": Skill(
    name="translate",
    description="Dịch văn bản Việt↔Anh. Chọn khi yêu cầu là DỊCH.",  # router đọc mô tả này
    persona="Bạn là biên dịch viên... chỉ trả về bản dịch.",
    tools=[],              # hoặc subset tools.TOOLS
    model="brain",         # hoặc "brain-pro" cho task khó
    max_turns=2,
),
```
Router tự nhận skill mới (không cần sửa `router.py`/`agent.py`). Nếu skill cần một model riêng
đã fine-tune (Bước 2 — multi-LoRA), đặt `model="brain-<name>"` khớp `model_name` trong
`config/litellm.yaml`.

## Mẹo với 7B (local)
Model local 7B hợp task **rõ ràng, phạm vi hẹp**: sửa 1 file, chạy 1 test, generate boilerplate.
Task lớn/mơ hồ hoặc review sâu → để skill route sang `--model brain-pro`, hoặc chia nhỏ.
