"""Eval matrix đa-task + CI gate (Bước 2 — docs/multitask-strategy.md).

Chạy một eval set (JSONL) qua gateway, tính metric PER-TASK, so với baseline, và áp
CỔNG CỨNG cho tool-calling (valid-JSON rate >= ngưỡng). In ma trận task×metric.

    # so adapter vs baseline
    python eval_matrix.py --eval data/eval.jsonl --model brain-extract --baseline brain

Eval set — mỗi dòng:
    {"task_id": "...", "messages": [...],   # phần prompt (system+user)
     "check": "json_valid" | "tool_call_valid" | "contains" | "regex",
     "expect": "<chuỗi/regex cho contains|regex>"}   # tùy check

Lõi đánh giá (evaluate) nhận complete_fn nên UNIT-TEST được không cần gateway.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Cổng cứng: task dùng check này phải đạt tối thiểu tỉ lệ pass.
HARD_GATES = {"tool_call_valid": 0.95}


# ---------------- Checkers (thuần, test được) ----------------
def check_json_valid(output: str, expect=None) -> bool:
    try:
        json.loads(output.strip())
        return True
    except (json.JSONDecodeError, AttributeError):
        return False


def check_tool_call_valid(output: str, expect=None) -> bool:
    """Có ít nhất 1 khối <tool_call> và MỌI khối parse được thành {name, arguments}."""
    blocks = TOOL_CALL_RE.findall(output or "")
    if not blocks:
        return False
    for b in blocks:
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            return False
        if "name" not in obj or "arguments" not in obj:
            return False
    return True


def check_contains(output: str, expect: str) -> bool:
    return bool(expect) and expect in (output or "")


def check_regex(output: str, expect: str) -> bool:
    return bool(expect) and re.search(expect, output or "") is not None


CHECKERS = {
    "json_valid": check_json_valid,
    "tool_call_valid": check_tool_call_valid,
    "contains": check_contains,
    "regex": check_regex,
}


def evaluate(items: list[dict], complete_fn, concurrency: int = 8) -> dict:
    """complete_fn(messages) -> str. Trả về {task: {check, n, pass, rate}} + raw."""
    def run(item):
        out = complete_fn(item["messages"])
        checker = CHECKERS[item["check"]]
        ok = checker(out, item.get("expect"))
        return item["task_id"], item["check"], ok

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(run, items))

    # Gom theo (task, check) — một task có thể có NHIỀU loại check, không được trộn chung.
    agg: dict = defaultdict(lambda: {"n": 0, "pass": 0})
    for task, check, ok in results:
        a = agg[(task, check)]
        a["n"] += 1
        a["pass"] += int(ok)
    out: dict = {}
    for (task, check), a in agg.items():
        out[(task, check)] = {
            "task": task, "check": check, "n": a["n"], "pass": a["pass"],
            "rate": a["pass"] / a["n"] if a["n"] else 0.0,
        }
    return out


def gate(matrix: dict) -> tuple[bool, list[str]]:
    """Cổng cứng: check nào có ngưỡng thì rate phải >= ngưỡng (áp cho TỪNG cặp task×check)."""
    fails = []
    for a in matrix.values():
        thr = HARD_GATES.get(a["check"])
        if thr is not None and a["rate"] < thr:
            fails.append(f"{a['task']}[{a['check']}] {a['rate']:.2%} < {thr:.0%}")
    return (len(fails) == 0), fails


def render(cand: dict, base: dict | None) -> str:
    lines = ["", f"{'task':12s} {'check':16s} {'n':>4s} {'cand':>7s} {'base':>7s} {'Δ':>7s}"]
    lines.append("-" * 58)
    for key in sorted(cand):                       # key = (task, check)
        c = cand[key]
        b = (base or {}).get(key)
        brate = f"{b['rate']:.2%}" if b else "  —  "
        delta = f"{(c['rate'] - b['rate']):+.2%}" if b else "  —  "
        lines.append(f"{c['task']:12s} {c['check']:16s} {c['n']:>4d} "
                     f"{c['rate']:>7.2%} {brate:>7s} {delta:>7s}")
    return "\n".join(lines)


def _gateway_complete_fn(model: str):
    """complete_fn thật, gọi gateway OpenAI-compatible."""
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("GATEWAY_URL", "http://localhost:4000/v1"),
        api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-change-me-please"),
    )

    def fn(messages):
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=1024)
        return r.choices[0].message.content or ""
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="Eval set JSONL")
    ap.add_argument("--model", required=True, help="Model/adapter cần chấm (vd brain-extract)")
    ap.add_argument("--baseline", default=None, help="Model nền để so (vd brain)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    for i, it in enumerate(items):
        if "messages" not in it or "task_id" not in it:
            raise ValueError(f"eval#{i}: thiếu 'messages' hoặc 'task_id'")
        if it.get("check") not in CHECKERS:
            raise ValueError(f"eval#{i}: check không hợp lệ: {it.get('check')}")

    cand = evaluate(items, _gateway_complete_fn(args.model), args.concurrency)
    base = evaluate(items, _gateway_complete_fn(args.baseline), args.concurrency) \
        if args.baseline else None

    print(render(cand, base))
    ok, fails = gate(cand)
    print("\nCI GATE:", "PASS ✔" if ok else "FAIL ✗ " + "; ".join(fails))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
