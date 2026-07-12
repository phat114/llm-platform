"""Sinh dữ liệu SFT bằng cách DISTILL từ model mạnh `brain-pro` (Claude) qua gateway.

Ý tưởng (docs/multitask-strategy.md — "Nguồn sinh nhanh"): lấy một tập prompt "seed"
của người dùng, gọi `brain-pro` qua gateway LiteLLM để sinh câu trả lời "vàng"
(assistant chất lượng cao, tiếng Việt), rồi xuất JSONL đúng schema train
(xem data/schema.md). Đây là nguồn distill tốt nhất trước khi trộn bằng prepare_data.py.

Định dạng file SEED (mỗi dòng 1 JSON):
    {"task_id": "extract", "user": "Tách ngày ra khỏi câu: 'hẹn 3/5 nhé'"}
    {"task_id": "chat",    "user": "Chào bạn", "system": "Bạn là trợ lý thân thiện."}
  - "task_id" (bắt buộc): khớp skill build/review/chat/extract... để cân bằng data-mix.
  - "user"    (bắt buộc): prompt người dùng.
  - "system"  (tùy chọn): nếu có thì đưa vào cả lời gọi distill LẪN mẫu train;
                nếu không có thì mẫu train chỉ gồm user + assistant.

Kết quả (mỗi dòng 1 JSON, đúng data/schema.md):
    {"task_id": "extract", "messages": [{"role":"system",...}?,
                                        {"role":"user",...},
                                        {"role":"assistant",...}]}

⚠️ TOOL-CALLING (task `build`/agent): script này KHÔNG tự thêm định nghĩa tool.
   Muốn khớp train ↔ serve (chat template Qwen2.5 chèn block `# Tools` vào system khi
   serve), phải TỰ nhúng đúng định nghĩa tool vào trường "system" của seed — xem mục
   "Giới hạn tool-calling & multi-turn" trong data/schema.md. Ở đây `system` seed sao
   thì mẫu train vậy, không chèn thêm gì.

Ví dụ:
    export LITELLM_MASTER_KEY=sk-...            # trùng .env của platform
    export GATEWAY_URL=http://SERVER_IP:4000/v1
    python distill_data.py --seed-file data/seed.jsonl --out data/distilled.jsonl \
        --model brain-pro --temperature 0.7 --concurrency 4 --limit 200
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} JSON lỗi: {e}") from e
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_client() -> OpenAI:
    """Client OpenAI trỏ gateway LiteLLM (giống agents/agent.py)."""
    base_url = os.environ.get("GATEWAY_URL", "http://localhost:4000/v1")
    api_key = os.environ.get("LITELLM_MASTER_KEY", "sk-change-me-please")
    return OpenAI(base_url=base_url, api_key=api_key)


def _validate_seed(row: dict, where: str) -> None:
    if "task_id" not in row or "user" not in row:
        raise ValueError(f"{where}: thiếu 'task_id' hoặc 'user'")
    if not str(row["user"]).strip():
        raise ValueError(f"{where}: 'user' rỗng")


def _dedup_key(user: str, assistant: str) -> str:
    """Hash(user + assistant) để loại mẫu trùng (tránh phồng data bằng bản sao)."""
    h = hashlib.sha256()
    h.update(user.encode("utf-8"))
    h.update(b"\x00")
    h.update(assistant.encode("utf-8"))
    return h.hexdigest()


def distill_one(client: OpenAI, seed: dict, *, model: str, temperature: float,
                max_tokens: int | None) -> dict:
    """Gọi gateway sinh câu trả lời 'vàng' cho 1 seed → mẫu train đúng schema.

    Ném exception nếu call lỗi để caller đếm + bỏ mẫu (không làm hỏng cả mẻ).
    """
    system = seed.get("system")
    user = str(seed["user"])

    # Dựng messages gửi lên gateway: nếu seed có system thì đưa vào ngữ cảnh distill.
    call_messages: list[dict] = []
    if system:
        call_messages.append({"role": "system", "content": str(system)})
    call_messages.append({"role": "user", "content": user})

    kwargs: dict = {"model": model, "messages": call_messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    assistant = (resp.choices[0].message.content or "").strip()
    if not assistant:
        raise ValueError("gateway trả về assistant rỗng")

    # Mẫu train: giữ đúng thứ tự system? → user → assistant (data/schema.md).
    out_messages: list[dict] = []
    if system:
        out_messages.append({"role": "system", "content": str(system)})
    out_messages.append({"role": "user", "content": user})
    out_messages.append({"role": "assistant", "content": assistant})
    return {"task_id": seed["task_id"], "messages": out_messages}


def distill(seeds: list[dict], *, model: str, temperature: float,
            max_tokens: int | None, concurrency: int) -> tuple[list[dict], int]:
    """Distill toàn bộ seeds song song. Trả (mẫu hợp lệ đã dedup, số mẫu bỏ)."""
    client = make_client()
    rows: list[dict] = []
    seen: set[str] = set()
    dropped = 0

    def _work(idx_seed: tuple[int, dict]):
        idx, seed = idx_seed
        return idx, seed, distill_one(
            client, seed, model=model, temperature=temperature, max_tokens=max_tokens,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_work, (i, s)) for i, s in enumerate(seeds)]
        for fut in as_completed(futures):
            try:
                idx, seed, row = fut.result()
            except Exception as e:  # noqa: BLE001 — 1 call lỗi không được làm hỏng cả mẻ
                dropped += 1
                print(f"  ⚠ bỏ 1 mẫu (call lỗi): {e}", file=sys.stderr)
                continue

            user = row["messages"][-2]["content"]
            assistant = row["messages"][-1]["content"]
            key = _dedup_key(user, assistant)
            if key in seen:
                dropped += 1
                print(f"  ⚠ bỏ 1 mẫu trùng (task={seed['task_id']})", file=sys.stderr)
                continue
            seen.add(key)
            rows.append(row)

    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Distill dữ liệu SFT từ brain-pro (Claude) qua gateway LiteLLM",
    )
    ap.add_argument("--seed-file", required=True, help="JSONL seed: {task_id,user,system?}")
    ap.add_argument("--out", default="data/distilled.jsonl", help="File JSONL kết quả")
    ap.add_argument("--model", default="brain-pro", help="Model distill (đè brain-pro)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=None, help="Trần token/câu trả lời")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ lấy N seed đầu")
    ap.add_argument("--concurrency", type=int, default=4, help="Số call song song")
    args = ap.parse_args()

    seeds = load_jsonl(args.seed_file)
    for i, s in enumerate(seeds):
        _validate_seed(s, f"{args.seed_file}#{i}")
    if args.limit is not None:
        seeds = seeds[: args.limit]

    rows, dropped = distill(
        seeds, model=args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_jsonl(args.out, rows)

    # Tóm tắt: số seed vào, số sinh thành công, số bỏ, đường dẫn out.
    print("\n=== TÓM TẮT DISTILL ===")
    print(f"  model       = {args.model}")
    print(f"  seed        = {len(seeds)}")
    print(f"  thành công  = {len(rows)}")
    print(f"  bỏ          = {dropped} (call lỗi + trùng)")
    print(f"  out         = {args.out}")


if __name__ == "__main__":
    main()
