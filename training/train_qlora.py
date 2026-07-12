"""Train MỘT adapter QLoRA (SFT) bằng Unsloth trên base bf16 Qwen2.5-Coder-7B.

Chạy trên GPU box (dev 3060 12GB dùng QLoRA ~5–9GB; server 24–48GB có thể tắt 4-bit).
    python train_qlora.py --config configs/extract.yaml

Điểm mấu chốt (khớp docs/multitask-strategy.md):
  - Train trên BASE bf16 (KHÔNG *-AWQ). Adapter xong đem serve trên vLLM (multi-LoRA).
  - Giữ ĐÚNG chat template Qwen2.5 (apply_chat_template) — giữ format tool-call hermes.
  - train_on_responses_only: mask system/user, chỉ tính loss phần assistant.
"""
from __future__ import annotations
import argparse
import inspect

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["adapter_name", "base_model", "output_dir", "data_files"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config thiếu khóa: {missing}")
    if "awq" in str(cfg["base_model"]).lower():
        raise ValueError("base_model là bản AWQ — KHÔNG train được. Dùng bản bf16 gốc.")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Import nặng đặt trong main để `--help`/load_config test được không cần GPU.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # 1) Base bf16 + QLoRA 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=None,                      # tự chọn theo GPU
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 3407),
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # 2) Data đã trộn (prepare_data.py). Lọc theo task nếu cấu hình.
    ds = load_dataset("json", data_files=cfg["data_files"], split="train")
    task_filter = cfg.get("task_filter")
    if task_filter:
        ds = ds.filter(lambda ex: ex.get("task_id") == task_filter)
    if len(ds) == 0:
        raise ValueError("Dataset rỗng sau khi lọc — kiểm tra data_files/task_filter.")

    # LƯU Ý (tool-calling): mẫu build ở data/example.jsonl nhúng khối <tool_call> thô và KHÔNG
    # truyền tools= vào apply_chat_template. Khi serve, agent gửi kèm định nghĩa tool → template
    # chèn block "# Tools" vào system. Để train/serve KHÔNG lệch, mẫu tool-calling nên nhúng đúng
    # định nghĩa tool vào system prompt (hoặc dùng messages có tool_calls + tools=). Xem data/schema.md.
    def to_text(ex: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    # 3) Trainer — chọn tên tham số theo version TRL (tránh crash do đổi API).
    #    TRL mới đổi SFTConfig.max_seq_length -> max_length; và SFTTrainer.tokenizer -> processing_class.
    seq_key = "max_length" if "max_length" in SFTConfig.__dataclass_fields__ else "max_seq_length"
    sft_args = SFTConfig(**{
        "dataset_text_field": "text",
        seq_key: cfg["max_seq_length"],
        "per_device_train_batch_size": cfg.get("per_device_batch_size", 1),
        "gradient_accumulation_steps": cfg.get("grad_accum", 8),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "num_train_epochs": cfg.get("epochs", 3),
        "learning_rate": float(cfg.get("learning_rate", 2e-4)),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "seed": cfg.get("seed", 3407),
        "logging_steps": cfg.get("logging_steps", 5),
        "output_dir": f"{cfg['output_dir']}/_run",
        "report_to": "none",
    })
    tok_key = ("processing_class"
               if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
               else "tokenizer")
    trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_args, **{tok_key: tokenizer})

    # 4) Mask prompt — chỉ học phần assistant (ChatML Qwen2.5).
    #    RÀNG BUỘC: chỉ đúng cho hội thoại SINGLE-TURN (1 user -> 1 assistant). Với multi-turn có
    #    role "tool"/nhiều lượt, vùng tool-result KHÔNG được mask bằng 1 instruction_part -> sẽ bị
    #    tính loss. Nếu cần multi-turn tool-calling, truyền instruction_part dạng list mọi header
    #    không-phải-assistant (kiểm token thật từ get_chat_template). Xem data/schema.md.
    if cfg.get("train_on_responses_only", True):
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    trainer.train()

    # 5) Lưu adapter (LoRA) — đem đi serve multi-LoRA.
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\n✔ Adapter '{cfg['adapter_name']}' đã lưu ở {cfg['output_dir']}")
    print("  Serve: thêm vào --lora-modules của vLLM + model_name trong litellm.yaml (xem README).")


if __name__ == "__main__":
    main()
