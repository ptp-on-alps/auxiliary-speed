#!/usr/bin/env python3
"""
Quality check: P(same token | quantile coupling) between a quantized model and a BF16 reference.

Usage:
  python quality_check.py --model swiss-ai/Apertus-70B-2509 \
                           --compressed-model RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16
"""

import argparse

import torch
from transformers import AutoTokenizer

import bench_common as bc


def get_logits(model, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits[0].cpu().float()  # [seq_len, vocab]


def report(quant_logits: torch.Tensor, ref_logits: torch.Tensor, label: str):
    p = torch.softmax(ref_logits, dim=-1)
    q = torch.softmax(quant_logits, dim=-1)
    cdf_p = p.cumsum(-1)
    cdf_q = q.cumsum(-1)
    zeros = torch.zeros(p.shape[0], 1)
    cdf_p_prev = torch.cat([zeros, cdf_p[:, :-1]], dim=-1)
    cdf_q_prev = torch.cat([zeros, cdf_q[:, :-1]], dim=-1)
    overlap = (torch.minimum(cdf_p, cdf_q) - torch.maximum(cdf_p_prev, cdf_q_prev)).clamp(min=0)
    prob_same = overlap.sum(-1).mean().item()
    print(f"\n  {label}")
    print(f"    P(same token | quantile coupling): {prob_same:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="P(same token | quantile coupling) quality check")
    parser.add_argument("--model", type=str, required=True,
                        help="BF16 reference model (HuggingFace name or local path)")
    parser.add_argument("--compressed-model", type=str, required=True,
                        help="Pre-quantized model to compare against the reference")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for quality check (default: 512)")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}  |  GPUs: {n_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer.encode(bc._QUALITY_TEXT, add_special_tokens=True)
    if len(tokens) < args.seq_len:
        tokens = (tokens * ((args.seq_len // len(tokens)) + 2))[:args.seq_len]
    else:
        tokens = tokens[:args.seq_len]
    input_ids = torch.tensor([tokens], device="cuda")
    print(f"\n  Sequence: {args.seq_len} tokens  |  text: {tokenizer.decode(tokens[:16])}…")

    print("\n" + "=" * 72)
    print("  QUALITY CHECK  (P(same token | quantile coupling))")
    print("=" * 72)

    print(f"\n  Loading reference: {args.model} ...")
    ref_model = bc.load_model(args.model)
    ref_logits = get_logits(ref_model, input_ids)
    del ref_model; bc.flush()

    print(f"\n  Loading compressed: {args.compressed_model} ...")
    comp_model = bc.load_model(args.compressed_model)
    comp_logits = get_logits(comp_model, input_ids)
    del comp_model; bc.flush()

    report(comp_logits, ref_logits, f"{args.compressed_model} vs {args.model}")


if __name__ == "__main__":
    main()
