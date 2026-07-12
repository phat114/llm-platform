"""Test config loader của train_qlora (chỉ cần pyyaml, không cần torch/GPU)."""
import os

import pytest

pytest.importorskip("yaml")
import yaml

import train_qlora as tq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_config_reads_extract():
    cfg = tq.load_config(os.path.join(ROOT, "training", "configs", "extract.yaml"))
    assert cfg["adapter_name"] == "extract"
    assert cfg["lora_r"] == 32 and cfg["lora_alpha"] == 64


def test_load_config_rejects_awq(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({
        "adapter_name": "x", "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "output_dir": "./o", "data_files": ["d.jsonl"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        tq.load_config(str(p))


def test_load_config_requires_keys(tmp_path):
    p = tmp_path / "incomplete.yaml"
    p.write_text(yaml.safe_dump({"adapter_name": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        tq.load_config(str(p))
