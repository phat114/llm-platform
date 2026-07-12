"""Router — chọn Skill phù hợp cho task.

"Router phân loại rẻ = chính brain": một lần gọi model (mặc định local `brain`, rất rẻ)
với structured output để chọn tên skill từ registry. Có fallback nhiều tầng nếu endpoint
không hỗ trợ json_schema, hoặc khi model trả không hợp lệ → dùng DEFAULT_SKILL.
"""
from __future__ import annotations
import json

from openai import OpenAI

from skills import SKILLS, DEFAULT_SKILL


def _json_schema() -> dict:
    """response_format ép model chọn đúng một tên skill hợp lệ (enum)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "route",
            "schema": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": list(SKILLS.keys())},
                    "reason": {"type": "string"},
                },
                "required": ["skill"],
                "additionalProperties": False,
            },
        },
    }


def choose_skill(client: OpenAI, task: str, model: str = "brain") -> tuple[str, str]:
    """Trả về (skill_name, lý_do). Không bao giờ raise — luôn có fallback."""
    catalog = "\n".join(f"- {s.name}: {s.description}" for s in SKILLS.values())
    system = (
        "Bạn là bộ định tuyến. Chọn ĐÚNG MỘT skill phù hợp nhất cho yêu cầu người dùng, "
        'từ danh sách dưới. Trả về JSON: {"skill": "<tên>", "reason": "<ngắn gọn>"}.\n\n'
        f"Danh sách skill:\n{catalog}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    # Thử json_schema (ép enum) trước; nếu endpoint không hỗ trợ → thử json_object.
    for response_format in (_json_schema(), {"type": "json_object"}):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                response_format=response_format, temperature=0, max_tokens=200,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            skill = data.get("skill")
            if skill in SKILLS:
                return skill, data.get("reason", "")
        except Exception:  # noqa: BLE001 - endpoint/parse lỗi → thử cách kế / fallback
            continue
    return DEFAULT_SKILL, "(router không xác định được → dùng mặc định)"
