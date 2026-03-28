#!/usr/bin/env python3
"""
Quality check: average Wasserstein-1 distance between INT8/INT4 and BF16 output distributions.
"""

import argparse

import torch
from transformers import AutoTokenizer

import bench_common as bc


def run_quality_check(model_name: str, tokenizer, seq_len: int,
                      skip_int8: bool, skip_int4: bool):
    print("\n" + "=" * 72)
    print("  QUALITY CHECK  (P(same token | quantile coupling) vs BF16 reference)")
    print("=" * 72)

    # Build a fixed input from real text, truncated/repeated to seq_len
    tokens = tokenizer.encode(bc._QUALITY_TEXT, add_special_tokens=True)
    if len(tokens) < seq_len:
        tokens = (tokens * ((seq_len // len(tokens)) + 2))[:seq_len]
    else:
        tokens = tokens[:seq_len]
    input_ids = torch.tensor([tokens], device="cuda")
    print(f"\n  Sequence: {seq_len} tokens  |  text: {tokenizer.decode(tokens[:16])}…")

    def get_logits(model) -> torch.Tensor:
        with torch.no_grad():
            return model(input_ids).logits[0].cpu().float()  # [seq_len, vocab]

    def report(quant_logits: torch.Tensor, ref_logits: torch.Tensor, label: str):
        p = torch.softmax(ref_logits, dim=-1)    # [seq_len, vocab]
        q = torch.softmax(quant_logits, dim=-1)
        # P(same token | quantile coupling): for each token k, measure the overlap
        # between its CDF intervals under p and q, then sum over tokens.
        cdf_p = p.cumsum(-1)
        cdf_q = q.cumsum(-1)
        zeros = torch.zeros(p.shape[0], 1)
        cdf_p_prev = torch.cat([zeros, cdf_p[:, :-1]], dim=-1)
        cdf_q_prev = torch.cat([zeros, cdf_q[:, :-1]], dim=-1)
        overlap = (torch.minimum(cdf_p, cdf_q) - torch.maximum(cdf_p_prev, cdf_q_prev)).clamp(min=0)
        prob_same = overlap.sum(-1).mean().item()
        print(f"\n  {label}")
        print(f"    P(same token | quantile coupling): {prob_same:.4f}")

    print("\n  Loading BF16 reference ...")
    ref_model = bc.load_model_bf16_eval(model_name)
    ref_logits = get_logits(ref_model)
    del ref_model; bc.flush()

    if not skip_int8:
        print("\n  Loading INT8 ...")
        model = bc.load_model_int8(model_name)
        report(get_logits(model), ref_logits, "INT8 vs BF16")
        del model; bc.flush()

    if not skip_int4:
        print("\n  Loading INT4 ...")
        model = bc.load_model_int4(model_name)
        report(get_logits(model), ref_logits, "INT4 vs BF16")
        del model; bc.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Avg Wasserstein-1 distance between INT8/INT4 and BF16 output distributions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf",
                        help="HuggingFace model name or local path")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for quality check (default: 512)")
    parser.add_argument("--skip-int8", action="store_true",
                        help="Skip INT8 quality check")
    parser.add_argument("--skip-int4", action="store_true",
                        help="Skip INT4 quality check")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}  |  GPUs: {n_gpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_quality_check(args.model, tokenizer, args.seq_len,
                      skip_int8=args.skip_int8, skip_int4=args.skip_int4)


if __name__ == "__main__":
    main()
