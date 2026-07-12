"""Test cân bằng data-mix (training/prepare_data.py) — thuần stdlib, không cần GPU."""
import random

import pytest

import prepare_data as pd


def test_temperature_guard():
    with pytest.raises(ValueError):
        pd.target_counts({"a": 5}, 0.0, 0.5, 1, 5.0)


def test_max_share_cap():
    counts = {"a": 1000, "b": 10}
    t = pd.target_counts(counts, temperature=1.0, max_share=0.5,
                         min_per_task=1, upsample_cap=5.0)
    cap = round(0.5 * sum(counts.values()))
    assert all(v <= cap for v in t.values())


def test_temperature_lifts_small_task_share():
    counts = {"big": 100, "small": 10}
    prop_small = 10 / 110
    t = pd.target_counts(counts, temperature=3.0, max_share=1.0,
                         min_per_task=1, upsample_cap=100.0)
    share_small = t["small"] / sum(t.values())
    assert share_small > prop_small          # T>1 nâng task nhỏ


def test_resample_exact_k():
    rng = random.Random(0)
    rows = [{"i": i} for i in range(5)]
    assert len(pd.resample(rows, 3, rng)) == 3     # downsample
    assert len(pd.resample(rows, 12, rng)) == 12   # upsample (có thay thế)


def test_stratified_split_holds_out_per_task():
    rng = random.Random(0)
    by_task = {"a": [{"x": i} for i in range(10)],
               "b": [{"y": i} for i in range(10)]}
    train, val = pd.stratified_split(by_task, 0.2, rng)
    assert len(val) == 4 and len(train) == 16


def test_build_mix_smoke():
    rows = ([{"task_id": "a", "messages": [1]} for _ in range(6)]
            + [{"task_id": "b", "messages": [1]} for _ in range(2)])
    train, val, stats = pd.build_mix(
        rows, temperature=1.7, max_share=1.0, min_per_task=4, upsample_cap=5.0,
        replay_rows=[], replay_frac=0.0, val_ratio=0.25, seed=1)
    assert stats["source_counts"] == {"a": 6, "b": 2}
    assert len(train) + len(val) == sum(stats["target_counts"].values())
    assert len(val) >= 1                     # có hold-out


def test_build_mix_replay_added():
    rows = [{"task_id": "a", "messages": [1]} for _ in range(10)]
    replay = [{"task_id": "gen", "messages": [1]} for _ in range(20)]
    _, _, stats = pd.build_mix(
        rows, temperature=1.0, max_share=1.0, min_per_task=1, upsample_cap=5.0,
        replay_rows=replay, replay_frac=0.1, val_ratio=0.0, seed=1)
    assert "_replay" in stats["target_counts"]     # replay được trộn vào
