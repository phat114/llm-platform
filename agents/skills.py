"""SKILLS registry — Bước 1 chiến lược đa tác vụ (xem docs/multitask-strategy.md).

Mỗi Skill = persona (system prompt) + bộ tool + model route + (chỗ cho RAG collection).
Router (router.py) chọn skill theo task; agent.py dựng Agent từ skill được chọn.

Thêm một tác vụ mới = thêm MỘT Skill ở đây — KHÔNG train lại gì. Đây chính là cách
"phủ đa task không-train" trước khi phải đụng tới LoRA.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import tools


@dataclass(frozen=True)
class Skill:
    name: str                 # định danh; gợi ý trùng model_name ở litellm (brain-<name>)
    description: str          # DÙNG CHO ROUTER: nêu rõ KHI NÀO chọn skill này
    persona: str              # system prompt riêng cho skill
    tools: list[Any] = field(default_factory=list)   # subset của tools.TOOLS
    model: str = "brain"      # route: "brain" (local GPU) / "brain-pro" (Claude) cho task khó
    max_turns: int = 12       # trần vòng lặp; task nhẹ để thấp, agent build cao hơn
    collection: str | None = None   # (Bước 2 / RAG) tên collection tri thức — CHƯA wire


# ---------------- Persona (system prompt) ----------------
BUILD_PERSONA = """Bạn là kỹ sư phần mềm tự động. Bạn có tool: read_file, write_file,
list_dir, run_shell. Hoàn thành yêu cầu theo các bước nhỏ:
1. Khảo sát (list_dir / read_file) TRƯỚC khi sửa — không đoán nội dung file.
2. Sửa/tạo file bằng write_file.
3. Chạy build/test bằng run_shell để tự kiểm chứng.
4. Nếu lỗi: đọc log, sửa lại, chạy lại. Khi xong, tóm tắt ngắn gọn việc đã làm.
Trả lời bằng tiếng Việt."""

REVIEW_PERSONA = """Bạn là chuyên gia review code (chất lượng, bảo mật, hiệu năng, thiết kế).
Bạn CHỈ ĐỌC — không sửa file, không chạy lệnh thay đổi hệ thống. Dùng list_dir/read_file
khảo sát code liên quan TRƯỚC khi nhận xét (không đoán). Báo cáo theo mức ưu tiên
(nghiêm trọng → nhỏ); mỗi vấn đề nêu: vị trí (file:dòng), lý do, và đề xuất sửa.
Trả lời bằng tiếng Việt, súc tích."""

CHAT_PERSONA = """Bạn là trợ lý tiếng Việt hữu ích: trả lời trực tiếp, súc tích, chính xác.
Bạn KHÔNG có công cụ thao tác file — chỉ giải thích/tư vấn/trò chuyện. Nếu yêu cầu thực chất
cần đọc/sửa file hay chạy lệnh, hãy nhắc người dùng chạy lại với: --skill build."""

EXTRACT_PERSONA = """Bạn là bộ trích xuất dữ liệu. Đọc văn bản người dùng cung cấp và trả về
DỮ LIỆU CÓ CẤU TRÚC dạng JSON hợp lệ, KHÔNG kèm giải thích thừa. Field thiếu thông tin thì
để null. Chỉ xuất JSON."""


# ---------------- Registry ----------------
SKILLS: dict[str, Skill] = {
    "build": Skill(
        name="build",
        description=("Tự động hoá lập trình nhiều bước: đọc/sửa file, chạy build/test/git "
                     "trong dự án. Chọn khi yêu cầu ĐỘNG TỚI FILE hoặc CẦN CHẠY LỆNH."),
        persona=BUILD_PERSONA,
        tools=tools.TOOLS + [tools.search_codebase],   # 4 tool + RAG (nếu đã index)
        model="brain",
        max_turns=20,
        collection="codebase",             # (Bước 2) dành cho RAG — chưa bật
    ),
    "review": Skill(
        name="review",
        description=("Rà soát/review code sâu: tìm lỗi, bảo mật, thiết kế. CHỈ ĐỌC, không sửa. "
                     "Chọn khi cần nhận xét/đánh giá/kiểm tra code có vấn đề gì."),
        persona=REVIEW_PERSONA,
        tools=[tools.read_file, tools.list_dir, tools.search_codebase],   # read-only + RAG
        model="brain-pro",                 # review sâu → route Claude (cần ANTHROPIC_API_KEY)
        max_turns=12,
        collection="codebase",
    ),
    "chat": Skill(
        name="chat",
        description=("Hỏi đáp/giải thích/trò chuyện tiếng Việt chung, KHÔNG cần đụng file hay "
                     "chạy lệnh. Đây là lựa chọn mặc định khi không chắc."),
        persona=CHAT_PERSONA,
        tools=[],
        model="brain",
        max_turns=4,
    ),
    "extract": Skill(
        name="extract",
        description=("Trích xuất/phân loại dữ liệu có cấu trúc (JSON) từ văn bản. Chọn khi cần "
                     "output JSON hoặc các field cố định."),
        persona=EXTRACT_PERSONA,
        tools=[],
        model="brain",
        max_turns=2,
    ),
}

# Fallback an toàn (không tool) khi router không chắc chắn.
DEFAULT_SKILL = "chat"
