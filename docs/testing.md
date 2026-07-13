# Test tay tool và skill

Quy trình test **bằng tay** — nhìn tận mắt tool chạy, thay vì đoán. Ba tầng, mỗi tầng cô lập một
loại lỗi khác nhau. **Đừng lên tầng sau khi tầng trước còn đỏ** — lỗi tầng dưới sẽ giả trang thành
lỗi tầng trên và bạn sẽ đi sửa nhầm chỗ.

| Tầng | Cô lập lỗi gì | Cần gateway? | Cần GPU? |
|---|---|---|---|
| 0. Tool trần | Tool đọc/ghi sai chỗ, cổng bảo mật thủng | không | không |
| 1. Schema model nhìn thấy | Model chọn sai tool vì mô tả tồi | không | không |
| 2. Router chọn skill | Task bị phân loại nhầm skill | có | có |
| 3. Agent chạy thật | Vòng lặp tool-calling, chất lượng model | có | có |

## Chuẩn bị

```bash
cd agents
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# Linux/macOS: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

export AGENT_WORKDIR=/duong/dan/repo-nhap        # BẮT BUỘC — dùng repo NHÁP, không phải repo thật
export GATEWAY_URL=http://localhost:4000/v1      # chỉ cần từ tầng 2
export LITELLM_MASTER_KEY=sk-...                 # lấy từ .env của platform
```

`AGENT_WORKDIR` được đọc **lúc import** — set nó *trước* khi chạy, không set sau. Chưa set thì tool
thao tác vào thư mục hiện tại, và bạn sẽ ghi bậy vào chính repo này.

---

## Tầng 0 — tool trần (không model, không GPU)

Tool **không gọi trực tiếp được**: `@function_tool` biến chúng thành object `FunctionTool`, nên
`tools.read_file("a.txt")` ném `TypeError: 'FunctionTool' object is not callable`. Dùng
[agents/trycall.py](../agents/trycall.py) — nó lo phần async + `ToolContext` + chuỗi JSON, và
**dịch lại lỗi bị SDK nuốt** thành thông báo đọc được.

```bash
python trycall.py write_file '{"path": "a.txt", "content": "hi"}'
python trycall.py read_file  '{"path": "a.txt"}'          # → hi
python trycall.py list_dir   '{"path": "."}'
```

### Ca quan trọng nhất: cổng bảo mật

`_safe_path()` là thứ **duy nhất** ngăn agent ghi file ra ngoài `AGENT_WORKDIR`. Phải tự tay xác nhận
nó còn sống:

```bash
python trycall.py read_file '{"path": "../../../etc/passwd"}'
python trycall.py read_file '{"path": "C:/Windows/win.ini"}'
```

Cả hai **phải** ra `✔ BỊ CHẶN (đúng — cổng an toàn hoạt động)`. Ra được nội dung file là thủng.

> Vì sao cần `trycall.py` cho việc này: `on_invoke_tool` của SDK **nuốt exception** thành chuỗi
> `"An error occurred while running the tool..."`. Gọi tay kiểu thường, bạn sẽ thấy một câu báo lỗi
> chung chung và tưởng tool hỏng — trong khi cổng bảo mật đang làm **đúng** việc của nó.

### Cổng xác nhận shell

```bash
python trycall.py run_shell '{"command": "echo hi"}'          # hiện prompt: chạy? `echo hi` (y/N)
python trycall.py run_shell '{"command": "echo hi"}' --yes    # bỏ hỏi → exit=0
```

Gõ `n` → `"Người dùng từ chối chạy lệnh."`. Đây chính là cổng mà cờ `--auto` của agent tắt đi.

### RAG chưa cài

```bash
python trycall.py search_codebase '{"query": "login"}'
# → RAG chưa cài. Cần: pip install fastembed chromadb, rồi index: python rag.py index
```

Ra traceback hay `"No module named 'chromadb'"` là sai — model không hành động được với câu đó.

---

## Tầng 1 — schema mà model thực sự nhìn thấy

```bash
python trycall.py --list
```

In ra tên, description (SDK tự sinh từ docstring) và `params_json_schema` của từng tool — **đúng
nguyên văn thứ được nhét vào prompt**.

Đây là tầng hay bị bỏ qua nhất, mà lại quyết định nhiều nhất. Model **không** biết tool là gì; nó chỉ
so câu của người dùng với các đoạn mô tả này rồi chọn cái giống nhất. **Mô tả tool chính là logic chọn
tool** — không có tầng nào khác. Model chọn sai tool thì sửa mô tả, đừng đi sửa code.

Đọc `--list` và tự hỏi:
- Mô tả có nói **KHI NÀO** dùng không, hay chỉ nói tool **LÀM GÌ**? ("Đọc nội dung file" là mô tả tồi;
  "Dùng khi đã biết đường dẫn. KHÔNG dùng để tìm kiếm — dùng `search_codebase`" mới là mô tả tốt.)
- Có hai tool nào **cùng đúng** cho một câu hỏi không? Nếu có, model sẽ tung đồng xu.
- Số tool có đang phình ra không? 3-5 tool thì model nhỏ còn phân biệt được; 15-20 là mô tả nhoè
  vào nhau. Đây là lý do lớn nhất khiến agent "ngu đi" khi bạn thêm tool.

Kiểm bộ tool của từng skill:

```bash
python trycall.py --skill review    # → read_file, list_dir, search_codebase  (KHÔNG write/shell)
python trycall.py --skill chat      # → (không tool)
```

Hai bất biến phải giữ: `review` **read-only**, còn `chat`/`extract` **0 tool**. Cái sau không chỉ là
thiết kế đẹp — [toolfix.py](../agents/toolfix.py) dựa vào nó để không vá nhầm JSON mà `extract` xuất
ra (đúng thiết kế) thành tool call.

---

## Tầng 2 — router chọn skill (cần gateway)

```bash
python agent.py --list-skills      # không gọi model, chỉ in registry
```

Rồi chạy vài task mẫu và **chỉ nhìn dòng `Skill = ...`**:

| Task mẫu | Skill kỳ vọng |
|---|---|
| "Thêm test cho parse_date rồi chạy pytest" | `build` (động tới file / chạy lệnh) |
| "Xem app.py có lỗi bảo mật gì không" | `review` (nhận xét, không sửa) |
| "Giải thích LoRA là gì" | `chat` (không đụng file) |
| "Trích JSON tên + email từ đoạn văn này" | `extract` |

Router ép model chọn bằng `json_schema` + enum, nên nó **không thể** trả về tên skill lạ; sai lắm thì
rơi về `chat` (`DEFAULT_SKILL`). Muốn bỏ qua router: `--skill build`.

---

## Tầng 3 — agent chạy thật

```bash
mkdir /tmp/repo-nhap && cd /tmp/repo-nhap && git init && echo "# Test" > README.md
export AGENT_WORKDIR=/tmp/repo-nhap

cd -/agents
python agent.py --skill build "Đọc README.md rồi thêm dòng '# ok' vào cuối"
```

Quan sát ba thứ, theo thứ tự:

1. Các dòng `→ read_file(...)` / `→ write_file(...)` — **bằng chứng tool được gọi thật**. Không có dòng
   nào = agent chỉ chat suông, tool-calling hỏng.
2. Dòng `[toolfix]` — miếng vá đang hoạt động. Với model 3B đây là **bình thường**, không phải lỗi:
   model trả tool-call dạng ```json thay vì thẻ `<tool_call>` chuẩn hermes, và `toolfix` dựng lại.
3. **Nội dung file sau khi chạy** — `cat README.md`. Đừng tin dòng log; tin cái file.

### Hai điều đã đo được, ghi ra đây để bạn khỏi tưởng mình làm sai

**Đừng dùng `--auto` với model `brain` (3B).** Cờ đó tắt cổng xác nhận shell, và trong lần test thật
con 3B đã tự ý chạy `git init` → `git add` → `git push origin main`. `AGENT_WORKDIR` chỉ chặn thao tác
**file**, **không** chặn lệnh shell.

**Model 3B gọi đúng tool nhưng dùng tool ngu.** Nó từng ghi đè README thành đúng một dòng, xoá sạch
nội dung cũ, trong khi yêu cầu là *nối thêm vào cuối*. Đây là ranh giới cứng:

- *Gọi đúng tool* = bài toán **định dạng** → sửa được bằng code (đã sửa: `toolfix.py`).
- *Dùng tool cho khôn* = bài toán **trí tuệ** → 3B không có, và không miếng vá nào cho được.

Muốn agent thực sự làm được việc, đường ngắn nhất là cho skill `build` chạy `brain-pro` (như `review`
đang làm), hoặc đổi model local lớn hơn.

---

## Test tự động

`pytest tests/ -q` từ repo root. CI ([Jenkinsfile](../Jenkinsfile)) chạy trong venv chỉ có
`pytest` + `pyyaml`, nên test nào cần `openai`/`agents` phải `pytest.importorskip` — xem
[tests/test_toolfix.py](../tests/test_toolfix.py).
