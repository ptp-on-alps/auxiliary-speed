#!/usr/bin/env python3
"""
Pre-quantize a model to INT4 or INT8 and save the weights to disk.

Usage:
    python quantize.py --model meta-llama/Llama-2-7b-hf --bits 4
    python quantize.py --model meta-llama/Llama-2-7b-hf --bits 8

bench_common.py will automatically load from the saved path on subsequent runs.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from bench_common import quantized_save_path


def quantize_and_save(model_name: str, bits: int):
    save_path = quantized_save_path(model_name, bits)

    if (save_path / "config.json").exists():
        print(f"Quantized weights already exist at {save_path} — nothing to do.")
        return save_path

    print(f"Quantizing {model_name} to INT{bits} ...")

    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:  # bits == 8
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=".",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".")

    print(f"Saving to {save_path} ...")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Done. Saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize a HuggingFace model to INT4 or INT8."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--bits", type=int, choices=[4, 8], default=4,
        help="Quantization bits (4 or 8, default: 4)",
    )
    args = parser.parse_args()
    quantize_and_save(args.model, args.bits)


if __name__ == "__main__":
    main()
