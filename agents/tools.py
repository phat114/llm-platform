"""Tool cho agent build. Mỗi tool = schema (OpenAI format) + hàm thực thi.

An toàn: mọi thao tác file/shell bị giới hạn trong AGENT_WORKDIR.
"""
from __future__ import annotations
import os
import pathlib
import subprocess

# Mọi thao tác bị khóa trong thư mục này (mặc định: thư mục hiện tại)
WORKDIR = pathlib.Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()


def _safe_path(rel: str) -> pathlib.Path:
    """Chặn path traversal ra ngoài WORKDIR."""
    p = (WORKDIR / rel).resolve()
    if p != WORKDIR and WORKDIR not in p.parents:
        raise ValueError(f"Đường dẫn ngoài WORKDIR bị chặn: {rel}")
    return p


def read_file(path: str, max_bytes: int = 100_000) -> str:
    data = _safe_path(path).read_text(encoding="utf-8", errors="replace")
    return data[:max_bytes]


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Đã ghi {len(content)} ký tự vào {path}"


def list_dir(path: str = ".") -> str:
    p = _safe_path(path)
    items = [
        f"{'d' if e.is_dir() else 'f'} {e.relative_to(WORKDIR)}"
        for e in sorted(p.iterdir())
    ]
    return "\n".join(items) or "(trống)"


def run_shell(command: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        command, shell=True, cwd=str(WORKDIR),
        capture_output=True, text=True, timeout=timeout,
    )
    return (
        f"exit={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-8000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )


# ---- Schema gửi cho model (chuẩn OpenAI function calling) ----
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Đọc nội dung một file (đường dẫn tương đối WORKDIR).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Ghi/đè nội dung một file. Tự tạo thư mục cha nếu cần.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "Liệt kê file/thư mục trong một đường dẫn.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Chạy lệnh shell trong WORKDIR (build, test, git, ...).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]

DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_shell": run_shell,
}
