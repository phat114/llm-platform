"""(Đường B) Merge adapter LoRA vào base bf16 → 1 model độc lập.

Dùng khi muốn serve như model liền (latency thấp nhất, hết mismatch AWQ+LoRA) thay vì
multi-LoRA động. Sau bước này, RE-QUANTIZE sang AWQ để serve trên 3060 (xem README).

    python merge_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter ./adapters/extract --out ./models/brain-extract-merged

Chạy được cả trên CPU (device_map="cpu", cần ~32GB RAM cho 7B) nếu không đủ VRAM.
"""
from __future__ import annotations
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base bf16 (KHÔNG *-AWQ)")
    ap.add_argument("--adapter", required=True, help="Thư mục adapter LoRA")
    ap.add_argument("--out", required=True, help="Thư mục model đã merge")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    args = ap.parse_args()

    if "awq" in args.base.lower():
        raise ValueError("Không merge trên bản AWQ — dùng base bf16 gốc.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()               # fuse LoRA vào trọng số base
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)

    print(f"✔ Model merged (bf16) lưu ở {args.out}")
    print("  Bước tiếp: re-quantize AWQ để serve trên 3060, ví dụ:")
    print("    - AutoAWQ: quantize --model", args.out, "→ awq  (calib 128–512 mẫu domain)")
    print("    - hoặc llm-compressor (chính chủ vLLM). Rồi trỏ MODEL trong .env sang bản AWQ,")
    print("      GIỮ SERVED_MODEL_NAME=brain và MODEL_AUTODETECT=false.")


if __name__ == "__main__":
    main()
