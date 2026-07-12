"""toolfix: vá tool-call model nhỏ trả về dạng text thành tool_calls chuẩn.

Các chuỗi trong TEST là output THẬT của Qwen2.5-Coder-3B-AWQ qua gateway, không phải bịa.
"""
import json

import pytest

toolfix = pytest.importorskip("toolfix", reason="cần openai SDK (agents/requirements.txt)")

VALID = {"read_file", "write_file", "list_dir", "run_shell"}


def _calls(content: str):
    """Chạy đúng đường mà _salvage đi: quét object → dựng tool_call."""
    out = []
    for obj in toolfix._candidates(content):
        call = toolfix._build_call(obj, VALID, len(out))
        if call is not None:
            out.append((call.function.name, json.loads(call.function.arguments)))
    return out


def test_markdown_fence():
    """Dạng model 3B thực sự trả về: khối ```json thay vì thẻ <tool_call>."""
    content = '```json\n{"name": "read_file", "arguments": {"path": "README.md"}}\n```'
    assert _calls(content) == [("read_file", {"path": "README.md"})]


def test_nhieu_tool_call_trong_mot_luot():
    """Model nhỏ hay nhả cả chuỗi lệnh đã lên kế hoạch một lúc — phải bắt được HẾT."""
    content = (
        '```json\n{"name": "read_file", "arguments": {"path": "a.md"}}\n```\n'
        '```json\n{"name": "write_file", "arguments": {"path": "a.md", "content": "x"}}\n```'
    )
    assert _calls(content) == [
        ("read_file", {"path": "a.md"}),
        ("write_file", {"path": "a.md", "content": "x"}),
    ]


def test_hermes_va_json_tran():
    assert _calls('<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>') == [
        ("list_dir", {"path": "."})
    ]
    assert _calls('{"name": "list_dir", "arguments": {"path": "."}}') == [
        ("list_dir", {"path": "."})
    ]


def test_object_hong_cu_phap_khong_giet_object_hop_le():
    """Model 3B từng nhả `"x" + bien` (không phải JSON) ở call thứ 2 — call 1 vẫn phải sống."""
    content = (
        '{"name": "read_file", "arguments": {"path": "a.py"}}\n'
        '{"name": "write_file", "arguments": {"content": "x" + y}}'
    )
    assert _calls(content) == [("read_file", {"path": "a.py"})]


def test_khong_va_van_xuoi():
    assert _calls("Docker là một nền tảng container.") == []


def test_khong_va_tool_khong_khai():
    """Chốt an toàn: chỉ vá tool ĐÃ KHAI trong request. Skill `extract` khai 0 tool
    nên JSON nó xuất ra (đúng thiết kế) không bao giờ bị hiểu nhầm thành tool call."""
    assert _calls('{"name": "delete_everything", "arguments": {}}') == []
    assert [
        c for o in toolfix._candidates('{"name": "read_file", "arguments": {}}')
        if (c := toolfix._build_call(o, set(), 0))
    ] == []
