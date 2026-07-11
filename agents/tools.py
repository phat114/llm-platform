"""Tool cho agent build — dùng OpenAI Agents SDK.

Khác bản cũ: schema function-calling được SDK TỰ SINH từ type hint + docstring
(không còn khai báo JSON tay). Hàm an toàn (_safe_path, WORKDIR) giữ nguyên.

An toàn: mọi thao tác file/shell bị giới hạn trong AGENT_WORKDIR.
"""
from __future__ import annotations
import os
import pathlib
import subprocess

from agents import function_tool

# Mọi thao tác bị khóa trong thư mục này (mặc định: thư mục hiện tại)
WORKDIR = pathlib.Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()

# Hỏi xác nhận trước khi chạy shell. agent.py đặt False khi có cờ --auto.
REQUIRE_SHELL_APPROVAL = True

# Giới hạn an toàn nội bộ — KHÔNG expose cho model (model không nên chỉnh).
MAX_READ_BYTES = 100_000
SHELL_TIMEOUT = 120
RAG_TOP_K = 5


def _safe_path(rel: str) -> pathlib.Path:
    """Chặn path traversal ra ngoài WORKDIR."""
    p = (WORKDIR / rel).resolve()
    if p != WORKDIR and WORKDIR not in p.parents:
        raise ValueError(f"Đường dẫn ngoài WORKDIR bị chặn: {rel}")
    return p


@function_tool
def read_file(path: str) -> str:
    """Đọc nội dung một file (đường dẫn tương đối WORKDIR).

    Args:
        path: Đường dẫn file, tương đối so với WORKDIR.
    """
    print(f"  → read_file({path})")
    data = _safe_path(path).read_text(encoding="utf-8", errors="replace")
    return data[:MAX_READ_BYTES]


@function_tool
def write_file(path: str, content: str) -> str:
    """Ghi/đè nội dung một file. Tự tạo thư mục cha nếu cần.

    Args:
        path: Đường dẫn file, tương đối so với WORKDIR.
        content: Nội dung ghi vào file.
    """
    print(f"  → write_file({path})")
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Đã ghi {len(content)} ký tự vào {path}"


@function_tool
def list_dir(path: str = ".") -> str:
    """Liệt kê file/thư mục trong một đường dẫn.

    Args:
        path: Đường dẫn thư mục, tương đối so với WORKDIR (mặc định gốc).
    """
    print(f"  → list_dir({path})")
    p = _safe_path(path)
    items = [
        f"{'d' if e.is_dir() else 'f'} {e.relative_to(WORKDIR)}"
        for e in sorted(p.iterdir())
    ]
    return "\n".join(items) or "(trống)"


@function_tool
def run_shell(command: str) -> str:
    """Chạy lệnh shell trong WORKDIR (build, test, git, ...).

    Args:
        command: Lệnh shell cần chạy.
    """
    print(f"  → run_shell({command!r})")
    # Cổng an toàn: hỏi trước khi chạy shell (trừ khi agent.py bật --auto)
    if REQUIRE_SHELL_APPROVAL:
        if input(f"    chạy? `{command}` (y/N) ").strip().lower() != "y":
            print("    (bỏ qua)")
            return "Người dùng từ chối chạy lệnh."
    proc = subprocess.run(
        command, shell=True, cwd=str(WORKDIR),
        capture_output=True, text=True, timeout=SHELL_TIMEOUT,
    )
    return (
        f"exit={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-8000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )


@function_tool
def search_codebase(query: str) -> str:
    """Tìm đoạn code/tài liệu liên quan trong WORKDIR bằng RAG (embedding + vector search).
    Dùng để ĐỊNH VỊ nhanh trước khi read_file — thay vì mò list_dir/read_file.

    Args:
        query: Nội dung cần tìm (mô tả chức năng, tên hàm, khái niệm...).
    """
    print(f"  → search_codebase({query!r})")
    try:
        import rag
    except ImportError:
        return ("RAG chưa cài. Cần: pip install fastembed chromadb, rồi index: python rag.py index")
    hits = rag.search(query, RAG_TOP_K)
    if not hits:
        return "(không tìm thấy — đã index chưa? chạy: python rag.py index)"
    return "\n\n".join(
        f"[{h['path']}] score={h['score']}\n{h['snippet']}" for h in hits)


# SDK nhận trực tiếp các FunctionTool cơ bản; search_codebase gắn theo skill (skills.py).
TOOLS = [read_file, write_file, list_dir, run_shell]
