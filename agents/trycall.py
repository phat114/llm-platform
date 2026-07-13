"""Gọi tay từng tool — KHÔNG cần model, GPU hay gateway. Dùng để test tay (docs/testing.md).

Vì sao cần file này? Vì tool KHÔNG gọi trực tiếp được:

    >>> import tools
    >>> tools.read_file("README.md")
    TypeError: 'FunctionTool' object is not callable

@function_tool (Agents SDK) biến hàm thành object FunctionTool. Đường gọi hợp lệ duy nhất là
`await tool.on_invoke_tool(ToolContext(...), '{"path": "..."}')` — async, và nhận CHUỖI JSON
chứ không phải kwargs. Tệ hơn: nó NUỐT exception thành chuỗi "An error occurred...", nên khi
_safe_path chặn path traversal (đúng!), người test tay lại tưởng tool hỏng. File này lo hết,
và dịch lại lỗi bị nuốt thành thông báo đọc được.

Dùng:
    export AGENT_WORKDIR=/duong/dan/repo-nhap      # BẮT BUỘC — nếu không sẽ ghi vào thư mục hiện tại

    python trycall.py --list                       # in tool + schema ĐÚNG NHƯ MODEL NHÌN THẤY
    python trycall.py --skill review               # in bộ tool của một skill
    python trycall.py read_file  '{"path": "README.md"}'
    python trycall.py write_file '{"path": "a.txt", "content": "hi"}'
    python trycall.py run_shell  '{"command": "echo hi"}' --yes    # --yes: bỏ hỏi xác nhận
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys

# Console Windows là cp1252 → mọi output tiếng Việt sẽ ném UnicodeEncodeError. Ép UTF-8 trước.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from agents.tool_context import ToolContext  # noqa: E402

import tools  # noqa: E402
from skills import SKILLS  # noqa: E402

# search_codebase KHÔNG nằm trong tools.TOOLS (nó được gắn theo skill trong skills.py)
ALL_TOOLS = {t.name: t for t in tools.TOOLS + [tools.search_codebase]}

# Chuỗi mà _safe_path ném ra khi chặn — SDK nuốt nó thành "An error occurred..."
BLOCKED_MARK = "Đường dẫn ngoài WORKDIR bị chặn"


def _print_list() -> None:
    """In ĐÚNG thứ model nhìn thấy: tên + description + JSON schema (SDK tự sinh từ docstring).

    Mô tả tool CHÍNH LÀ logic chọn tool — model không có tầng nào khác để quyết định.
    Model chọn sai tool thì sửa ở đây, không phải sửa code.
    """
    for t in ALL_TOOLS.values():
        print(f"\n=== {t.name} ===")
        print(f"description: {t.description}")
        print("params_json_schema:")
        print(json.dumps(t.params_json_schema, ensure_ascii=False, indent=2))


def _print_skill(name: str) -> None:
    skill = SKILLS[name]
    tnames = [getattr(t, "name", "?") for t in skill.tools] or ["(không tool)"]
    print(f"skill  : {skill.name}")
    print(f"model  : {skill.model}   max_turns: {skill.max_turns}")
    print(f"tools  : {', '.join(tnames)}")
    print(f"mô tả  : {skill.description}")


def _invoke(tool_name: str, args_json: str) -> str:
    tool = ALL_TOOLS[tool_name]
    ctx = ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id="manual",
        tool_arguments=args_json,
    )
    return asyncio.run(tool.on_invoke_tool(ctx, args_json))


def main() -> None:
    ap = argparse.ArgumentParser(description="Gọi tay tool của agent (không cần model)")
    ap.add_argument("tool", nargs="?", choices=list(ALL_TOOLS), help="Tên tool cần gọi")
    ap.add_argument("args", nargs="?", default="{}", help="Tham số dạng JSON")
    ap.add_argument("--list", action="store_true", help="In tool + schema model nhìn thấy")
    ap.add_argument("--skill", choices=list(SKILLS), help="In bộ tool của một skill")
    ap.add_argument("--yes", action="store_true", help="run_shell: bỏ qua hỏi xác nhận")
    a = ap.parse_args()

    if not os.environ.get("AGENT_WORKDIR"):
        print(
            f"CẢNH BÁO: chưa set AGENT_WORKDIR → tool sẽ thao tác trong {tools.WORKDIR}\n"
            f"         Đặt nó trỏ vào một repo NHÁP trước khi test ghi file.\n",
            file=sys.stderr,
        )

    if a.list:
        _print_list()
        return
    if a.skill:
        _print_skill(a.skill)
        return
    if not a.tool:
        ap.error("thiếu tên tool (hoặc dùng --list / --skill)")

    if a.yes:
        tools.REQUIRE_SHELL_APPROVAL = False

    try:
        json.loads(a.args)
    except ValueError as e:
        ap.error(f"tham số không phải JSON hợp lệ: {e}")

    print(f"WORKDIR = {tools.WORKDIR}")
    result = _invoke(a.tool, a.args)

    # SDK nuốt mọi exception thành chuỗi. Dịch ngược lại cho người đọc.
    print("\n--- kết quả ---")
    if BLOCKED_MARK in result:
        print(f"✔ BỊ CHẶN (đúng — cổng an toàn hoạt động)\n  {result}")
    elif result.startswith("An error occurred"):
        print(f"✘ TOOL LỖI (SDK đã nuốt exception gốc, chỉ còn chuỗi này)\n  {result}")
    else:
        print(result)


if __name__ == "__main__":
    main()
