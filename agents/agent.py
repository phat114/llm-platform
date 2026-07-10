"""Agent đa-kỹ-năng trên OpenAI Agents SDK + SKILLS registry.

Luồng: task → router chọn skill (persona + tool + model route) → dựng Agent → chạy.
Thêm một tác vụ mới = thêm 1 Skill trong skills.py (KHÔNG train). Xem docs/multitask-strategy.md.

Ví dụ:
    export LITELLM_MASTER_KEY=sk-...           # trùng .env của platform
    export GATEWAY_URL=http://SERVER_IP:4000/v1
    export AGENT_WORKDIR=/duong/dan/project    # thư mục agent được phép thao tác

    python agent.py "Thêm test cho parse_date trong utils.py rồi chạy pytest"
    python agent.py --skill review "Xem app.py có lỗi bảo mật gì không"
    python agent.py --list-skills

Cờ:
    --skill NAME    ép skill, bỏ qua router
    --model NAME    đè model của skill (brain = local 3060, brain-pro = API ngoài)
    --max-steps N   đè max_turns của skill
    --auto          chạy shell KHÔNG hỏi xác nhận (dùng khi tin tưởng task)
    --list-skills   in registry rồi thoát
"""
from __future__ import annotations
import argparse
import os

from openai import AsyncOpenAI, OpenAI
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)

import tools
import router
from skills import SKILLS


def _print_skills() -> None:
    print("Skills khả dụng:")
    for s in SKILLS.values():
        tnames = ", ".join(getattr(t, "name", "?") for t in s.tools) or "(không tool)"
        rag = f", RAG={s.collection}(chưa bật)" if s.collection else ""
        print(f"  - {s.name:8s} [model={s.model}, max_turns={s.max_turns}{rag}]")
        print(f"      tools: {tnames}")
        print(f"      {s.description}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="Yêu cầu cho agent")
    ap.add_argument("--skill", choices=list(SKILLS.keys()), help="Ép skill, bỏ qua router")
    ap.add_argument("--model", default=None, help="Đè model của skill")
    ap.add_argument("--max-steps", type=int, default=None, help="Đè max_turns của skill")
    ap.add_argument("--auto", action="store_true", help="Chạy shell không cần xác nhận")
    ap.add_argument("--list-skills", action="store_true", help="In registry rồi thoát")
    args = ap.parse_args()

    if args.list_skills:
        _print_skills()
        return
    if not args.task:
        ap.error("thiếu 'task' (hoặc dùng --list-skills)")

    # Self-host: không gửi trace ra OpenAI (tránh cảnh báo thiếu OPENAI_API_KEY).
    set_tracing_disabled(True)
    # --auto → tool run_shell không hỏi xác nhận nữa
    tools.REQUIRE_SHELL_APPROVAL = not args.auto

    base_url = os.environ.get("GATEWAY_URL", "http://localhost:4000/v1")
    api_key = os.environ.get("LITELLM_MASTER_KEY", "sk-change-me-please")
    aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)   # cho Agents SDK (async)
    sclient = OpenAI(base_url=base_url, api_key=api_key)        # cho router (1-shot, sync)

    # 1) Chọn skill: ép bằng --skill, hoặc để router (chính brain) phân loại.
    if args.skill:
        skill_name, why = args.skill, "(ép bằng --skill)"
    else:
        skill_name, why = router.choose_skill(sclient, args.task)
    skill = SKILLS[skill_name]
    model_name = args.model or skill.model
    max_turns = args.max_steps if args.max_steps is not None else skill.max_turns

    print(f"WORKDIR = {tools.WORKDIR}")
    print(f"Skill   = {skill_name}  (model={model_name}, max_turns={max_turns})  {why}")

    # 2) Dựng Agent từ skill (persona + tool + model route).
    agent = Agent(
        name=skill.name,
        instructions=skill.persona,
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=aclient),
        model_settings=ModelSettings(temperature=0.2),
        tools=skill.tools,
    )

    # 3) Chạy.
    result = Runner.run_sync(agent, args.task, max_turns=max_turns)
    print(f"\n=== XONG ===\n{result.final_output}")


if __name__ == "__main__":
    main()
