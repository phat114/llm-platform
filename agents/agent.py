"""Agent build tối giản: vòng lặp tool-calling trỏ vào LLM Platform gateway.

Ví dụ:
    export LITELLM_MASTER_KEY=sk-...           # trùng .env của platform
    export AGENT_WORKDIR=/duong/dan/project    # thư mục agent được phép thao tác
    python agent.py "Đọc app.py, thêm test cho hàm parse_date rồi chạy pytest"

Cờ:
    --model brain       model dùng (brain = local 3060, brain-pro = API ngoài)
    --max-steps 25      số bước tối đa
    --auto              chạy shell KHÔNG hỏi xác nhận (dùng khi tin tưởng task)
"""
from __future__ import annotations
import argparse
import json
import os

from openai import OpenAI

import tools

SYSTEM = """Bạn là kỹ sư phần mềm tự động. Bạn có tool: read_file, write_file,
list_dir, run_shell. Hoàn thành yêu cầu theo các bước nhỏ:
1. Khảo sát (list_dir / read_file) TRƯỚC khi sửa — không đoán nội dung file.
2. Sửa/tạo file bằng write_file.
3. Chạy build/test bằng run_shell để tự kiểm chứng.
4. Nếu lỗi: đọc log, sửa lại, chạy lại. Khi xong, tóm tắt ngắn gọn việc đã làm.
Trả lời bằng tiếng Việt."""


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _short(d: dict, n: int = 140) -> str:
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="Yêu cầu cho agent")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "brain"))
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--auto", action="store_true",
                    help="Chạy shell không cần xác nhận")
    args = ap.parse_args()

    client = OpenAI(
        base_url=os.environ.get("GATEWAY_URL", "http://localhost:4000/v1"),
        api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-change-me-please"),
    )

    print(f"WORKDIR = {tools.WORKDIR}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.task},
    ]

    for step in range(1, args.max_steps + 1):
        resp = client.chat.completions.create(
            model=args.model, messages=messages,
            tools=tools.TOOLS, tool_choice="auto", temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # Không gọi tool nữa → agent coi như xong
        if not msg.tool_calls:
            print(f"\n=== XONG (bước {step}) ===\n{msg.content}")
            return

        for call in msg.tool_calls:
            name = call.function.name
            try:
                fn_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            print(f"\n[bước {step}] → {name}({_short(fn_args)})")

            # Cổng an toàn: hỏi trước khi chạy shell (trừ khi --auto)
            if name == "run_shell" and not args.auto:
                cmd = fn_args.get("command", "")
                if input(f"    chạy? `{cmd}` (y/N) ").strip().lower() != "y":
                    messages.append(_tool_result(call.id, "Người dùng từ chối chạy lệnh."))
                    print("    (bỏ qua)")
                    continue

            fn = tools.DISPATCH.get(name)
            if fn is None:
                result = f"LỖI: tool không tồn tại: {name}"
            else:
                try:
                    result = str(fn(**fn_args))
                except Exception as e:  # noqa: BLE001 - báo lỗi lại cho model tự sửa
                    result = f"LỖI: {e}"
            print(f"    ← {result[:300]}")
            messages.append(_tool_result(call.id, result))

    print("\n!! Đạt giới hạn bước, dừng.")


if __name__ == "__main__":
    main()
