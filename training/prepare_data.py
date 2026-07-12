"""Trộn & cân bằng dữ liệu đa-task cho SFT (Bước 2 — docs/multitask-strategy.md).

Chống negative transfer + quên tool-calling bằng:
  - temperature sampling p_i ∝ n_i^(1/T) (T>1 kéo task lớn xuống, nâng task nhỏ),
  - cap trần share cho task áp đảo (--max-share),
  - sàn tối thiểu + upsample task hiếm có giới hạn (--min-per-task, --upsample-cap),
  - replay % dữ liệu general/code gốc (--replay-file/--replay-frac) chống catastrophic forgetting,
  - split train/val PHÂN TẦNG theo task.

Thuần Python (json + random) — KHÔNG cần GPU/torch. Kết quả tất định theo --seed.

Ví dụ:
    python prepare_data.py --data data/example.jsonl \
        --out-dir data/mixed --temperature 1.7 --max-share 0.35 \
        --min-per-task 300 --val-ratio 0.1 --replay-file data/general.jsonl --replay-frac 0.1
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict


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


def _validate(row: dict, where: str) -> None:
    if "task_id" not in row or "messages" not in row:
        raise ValueError(f"{where}: thiếu 'task_id' hoặc 'messages'")
    if not isinstance(row["messages"], list) or not row["messages"]:
        raise ValueError(f"{where}: 'messages' phải là list không rỗng")


def target_counts(counts: dict[str, int], temperature: float,
                  max_share: float, min_per_task: int,
                  upsample_cap: float) -> dict[str, int]:
    """Số mẫu mục tiêu mỗi task sau khi cân bằng (giữ tổng ~ tổng gốc)."""
    if temperature <= 0:
        raise ValueError("temperature phải > 0 (T=1 giữ nguyên tỉ lệ; T>1 cân bằng task nhỏ)")
    total = sum(counts.values())
    # weight theo temperature: T=1 giữ nguyên tỉ lệ; T->inf tiến tới đều nhau.
    weights = {t: n ** (1.0 / temperature) for t, n in counts.items()}
    wsum = sum(weights.values())
    targets: dict[str, int] = {}
    for t, n in counts.items():
        raw = int(round(weights[t] / wsum * total))
        raw = max(raw, min(min_per_task, int(n * upsample_cap)))   # sàn, nhưng không vượt cap upsample
        raw = min(raw, int(n * upsample_cap))                      # trần upsample (chống lặp quá đà)
        raw = min(raw, int(round(max_share * total)))              # trần share tổng
        targets[t] = max(raw, 1)
    return targets


def resample(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Lấy k mẫu: không thay thế nếu k<=len, có thay thế (upsample) nếu k>len."""
    if k <= len(rows):
        return rng.sample(rows, k)
    out = list(rows)                        # giữ toàn bộ bản gốc trước
    out += [rng.choice(rows) for _ in range(k - len(rows))]
    return out


def stratified_split(by_task: dict[str, list[dict]], val_ratio: float,
                     rng: random.Random) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for t, rows in by_task.items():
        rows = rows[:]
        rng.shuffle(rows)
        n_val = int(round(len(rows) * val_ratio))
        val += rows[:n_val]
        train += rows[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_mix(rows: list[dict], *, temperature: float, max_share: float,
              min_per_task: int, upsample_cap: float, replay_rows: list[dict],
              replay_frac: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    counts = {t: len(v) for t, v in by_task.items()}

    targets = target_counts(counts, temperature, max_share, min_per_task, upsample_cap)
    mixed_by_task: dict[str, list[dict]] = {
        t: resample(by_task[t], targets[t], rng) for t in by_task
    }

    # Replay: thêm % dữ liệu general/code gốc để chống quên năng lực nền.
    if replay_rows and replay_frac > 0:
        cur_total = sum(len(v) for v in mixed_by_task.values())
        k = int(round(cur_total * replay_frac))
        if k > 0:
            mixed_by_task["_replay"] = resample(replay_rows, k, rng)

    train, val = stratified_split(mixed_by_task, val_ratio, rng)
    stats = {
        "source_counts": counts,
        "target_counts": {t: len(v) for t, v in mixed_by_task.items()},
        "train": len(train), "val": len(val),
        "params": {"temperature": temperature, "max_share": max_share,
                   "min_per_task": min_per_task, "upsample_cap": upsample_cap,
                   "replay_frac": replay_frac, "val_ratio": val_ratio, "seed": seed},
    }
    return train, val, stats


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trộn & cân bằng data đa-task cho SFT")
    ap.add_argument("--data", nargs="+", required=True, help="Một hoặc nhiều file JSONL")
    ap.add_argument("--out-dir", default="data/mixed")
    ap.add_argument("--temperature", type=float, default=1.7, help=">1 cân bằng task nhỏ")
    ap.add_argument("--max-share", type=float, default=0.35, help="Trần share/task")
    ap.add_argument("--min-per-task", type=int, default=300)
    ap.add_argument("--upsample-cap", type=float, default=5.0, help="Trần lặp task hiếm (x lần)")
    ap.add_argument("--replay-file", default=None, help="JSONL general/code gốc để replay")
    ap.add_argument("--replay-frac", type=float, default=0.1)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict] = []
    for p in args.data:
        for i, r in enumerate(load_jsonl(p)):
            _validate(r, f"{p}#{i}")
            rows.append(r)
    replay_rows = load_jsonl(args.replay_file) if args.replay_file else []

    train, val, stats = build_mix(
        rows, temperature=args.temperature, max_share=args.max_share,
        min_per_task=args.min_per_task, upsample_cap=args.upsample_cap,
        replay_rows=replay_rows, replay_frac=args.replay_frac,
        val_ratio=args.val_ratio, seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n✔ Đã ghi train/val/stats vào {args.out_dir}/")


if __name__ == "__main__":
    main()
