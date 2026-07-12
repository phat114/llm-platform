"""Vá tool-calling cho model nhỏ (Qwen2.5-Coder-3B-AWQ trên GPU 6GB).

VẤN ĐỀ THẬT (đo được, không phải giả định):
    Model 3B chọn ĐÚNG hàm và điền ĐÚNG tham số, nhưng trả về dưới dạng khối
    markdown ```json {...}``` thay vì bọc trong thẻ <tool_call> mà vLLM
    (--tool-call-parser hermes) cần. Parser không khớp → trường tool_calls rỗng →
    Agents SDK tưởng model đã trả lời xong → agent CHẲNG BAO GIỜ gọi tool, chỉ
    chat suông. Skill `build`/`review` coi như hỏng.

    Ép tool_choice="required" chữa được định dạng nhưng làm agent gọi tool ở MỌI
    lượt, kể cả lượt lẽ ra phải chốt câu trả lời → vòng lặp không thoát được.

CÁCH VÁ:
    Bọc AsyncOpenAI. Sau mỗi response, nếu message không có tool_calls nhưng
    content chứa JSON {"name": ..., "arguments": {...}} khớp một tool ĐÃ KHAI
    trong chính request đó → dựng lại tool_calls chuẩn.

    Ràng buộc "khớp tool đã khai" là thứ giữ cho miếng vá an toàn: skill `extract`
    xuất JSON thuần là đúng thiết kế, nhưng nó khai 0 tool → không có gì để khớp
    → không bị vá nhầm.

Bỏ miếng vá này đi khi đổi sang model đủ lớn để tuân thủ định dạng hermes.
"""
from __future__ import annotations

import json
import sys
from typing import Any

try:  # openai >= 2.x
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageFunctionToolCall as _ToolCall,
        Function as _Function,
    )
except ImportError:  # openai 1.x
    from openai.types.chat.chat_completion_message_tool_call import (  # type: ignore[no-redef]
        ChatCompletionMessageToolCall as _ToolCall,
        Function as _Function,
    )

_warned = False


def _candidates(content: str):
    """Mọi JSON object cấp cao nhất trong content, theo thứ tự xuất hiện.

    Quét bằng raw_decode thay vì regex, nên xử lý được HẾT các kiểu model 3B trả về:
    bọc <tool_call>, bọc ```json, JSON trần, và — quan trọng — NHIỀU object liền
    nhau trong một lượt (model nhỏ hay nhả cả chuỗi lệnh đã lên kế hoạch một lúc).
    Object nào hỏng cú pháp thì bỏ qua, các object hợp lệ còn lại vẫn dùng được.
    """
    decoder = json.JSONDecoder()
    i = 0
    while (i := content.find("{", i)) >= 0:
        try:
            obj, end = decoder.raw_decode(content, i)
        except ValueError:
            i += 1                          # không phải JSON hợp lệ → dịch qua
            continue
        if isinstance(obj, dict):
            yield obj
        i = end


def _declared_tool_names(kwargs: dict[str, Any]) -> set[str]:
    """Tên các tool đã khai trong request. Chỉ vá tool nằm trong tập này."""
    names: set[str] = set()
    for t in kwargs.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else getattr(t, "function", None)
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if name:
            names.add(name)
    return names


def _build_call(obj: dict[str, Any], valid: set[str], idx: int) -> Any | None:
    if obj.get("name") not in valid:
        return None

    args = obj.get("arguments", obj.get("parameters", {}))
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)   # tool_calls yêu cầu arguments là CHUỖI

    return _ToolCall(
        id=f"call_salvaged_{idx}",
        type="function",
        function=_Function(name=obj["name"], arguments=args),
    )


def _salvage(resp: Any, valid: set[str]) -> Any:
    global _warned
    if not valid:
        return resp

    for choice in getattr(resp, "choices", []):
        msg = getattr(choice, "message", None)
        if msg is None or msg.tool_calls or not msg.content:
            continue

        calls = []
        for obj in _candidates(msg.content):
            call = _build_call(obj, valid, len(calls))
            if call is not None:
                calls.append(call)
        if not calls:
            continue

        msg.tool_calls = calls
        msg.content = None
        choice.finish_reason = "tool_calls"

        if not _warned:
            _warned = True
            print(
                "  [toolfix] Model trả tool-call dạng text (không đúng chuẩn hermes) "
                "→ đã dựng lại thành tool_calls. Bình thường với model 3B.",
                file=sys.stderr,
            )
    return resp


# ---------- Lớp bọc client: chỉ chen vào chat.completions.create ----------

class _Completions:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        resp = await self._inner.create(*args, **kwargs)
        if kwargs.get("stream"):
            return resp                      # stream: để nguyên, không vá được theo chunk
        return _salvage(resp, _declared_tool_names(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.completions = _Completions(inner.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SalvagingClient:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat = _Chat(inner.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap(client: Any) -> Any:
    """Bọc AsyncOpenAI để tự vá tool-call trả về dạng text."""
    return _SalvagingClient(client)
