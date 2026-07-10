"""Agent build trên OpenAI Agents SDK, trỏ vào LLM Platform gateway.

Khác bản cũ: vòng lặp tool-calling tay được thay bằng Runner của SDK
(tự điều phối tool, tự retry cấu trúc). Logic tool/WORKDIR ở tools.py giữ nguyên.

Ví dụ:
    export LITELLM_MASTER_KEY=sk-...           # trùng .env của platform
    export GATEWAY_URL=http://SERVER_IP:4000/v1
    export AGENT_WORKDIR=/duong/dan/project    # thư mục agent được phép thao tác
    python agent.py "Đọc app.py, thêm test cho hàm parse_date rồi chạy pytest"

Cờ:
    --model brain       model dùng (brain = local 3060, brain-pro = API ngoài)
    --max-steps 25      trần số lượt (turn) của agent
    --auto              chạy shell KHÔNG hỏi xác nhận (dùng khi tin tưởng task)
"""
from __future__ import annotations
import argparse
import os

from openai import AsyncOpenAI
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)

import tools

SYSTEM = """Bạn là kỹ sư phần mềm tự động. Bạn có tool: read_file, write_file,
list_dir, run_shell. Hoàn thành yêu cầu theo các bước nhỏ:
1. Khảo sát (list_dir / read_file) TRƯỚC khi sửa — không đoán nội dung file.
2. Sửa/tạo file bằng write_file.
3. Chạy build/test bằng run_shell để tự kiểm chứng.
4. Nếu lỗi: đọc log, sửa lại, chạy lại. Khi xong, tóm tắt ngắn gọn việc đã làm.
Trả lời bằng tiếng Việt."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="Yêu cầu cho agent")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "brain"))
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--auto", action="store_true",
                    help="Chạy shell không cần xác nhận")
    args = ap.parse_args()

    # Self-host: không gửi trace ra OpenAI (tránh cảnh báo thiếu OPENAI_API_KEY).
    set_tracing_disabled(True)

    # --auto → tool run_shell không hỏi xác nhận nữa
    tools.REQUIRE_SHELL_APPROVAL = not args.auto

    # Client trỏ vào gateway của platform. Dùng Chat Completions (không phải
    # Responses API) vì endpoint là OpenAI-compatible qua LiteLLM/vLLM.
    client = AsyncOpenAI(
        base_url=os.environ.get("GATEWAY_URL", "http://localhost:4000/v1"),
        api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-change-me-please"),
    )

    agent = Agent(
        name="builder",
        instructions=SYSTEM,
        model=OpenAIChatCompletionsModel(model=args.model, openai_client=client),
        model_settings=ModelSettings(temperature=0.2),
        tools=tools.TOOLS,
    )

    print(f"WORKDIR = {tools.WORKDIR}")
    result = Runner.run_sync(agent, args.task, max_turns=args.max_steps)
    print(f"\n=== XONG ===\n{result.final_output}")


if __name__ == "__main__":
    main()
