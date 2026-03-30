#!/usr/bin/env python3
"""
Benchmark: Inference logit computation vs Training compute ratio
Target: 70B parameter model on 4x GH200 GPUs

Measures wall-clock time per token for:
  1. Quantized inference (INT8 and INT4) — full logit computation
  2. BF16 training — forward + backward + optimizer step

Usage:
  python inference_speed.py --model meta-llama/Llama-2-70b-hf
"""

import argparse
import json
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

import bench_common as bc


def main():
    parser = argparse.ArgumentParser(
        description="Logit vs Training compute ratio benchmark (fully optimized)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf",
                        help="HuggingFace model name or local path")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (keep small for 70B)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (extra added for torch.compile)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Timed iterations (median reported)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (useful for debugging)")
    parser.add_argument("--skip-int8", action="store_true",
                        help="Skip INT8 inference benchmark")
    parser.add_argument("--skip-int4", action="store_true",
                        help="Skip INT4 inference benchmark")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training benchmark")
    args = parser.parse_args()

    if args.no_compile:
        bc._compile_disabled = True

    print("=" * 72)
    print("  Logit Computation vs Training — Compute Ratio Benchmark")
    print("  (fully optimized: compile + flash attn + TF32 + fused optim)")
    print("=" * 72)

    # ── Environment info ──
    n_gpus = torch.cuda.device_count()
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPUs:    {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  "
              f"({props.total_memory / (1024**3):.1f} GB, "
              f"SM {props.major}.{props.minor})")

    print(f"\nOptimizations:")
    print(f"  TF32 matmul:       {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cuDNN benchmark:   {torch.backends.cudnn.benchmark}")
    print(f"  cuDNN TF32:        {torch.backends.cudnn.allow_tf32}")
    print(f"  torch.compile:     {not bc._compile_disabled}")
    print(f"  Flash Attention 2: True")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    vocab_size = tokenizer.vocab_size

    def make_input(bs, seq_len):
        return torch.randint(100, vocab_size - 100, (bs, seq_len), device="cuda")

    # ── INT4 Inference ──
    if not args.skip_int4:
        model = bc.load_model_int4(args.model)
        n_gpus_model = bc.count_gpus_used(model)
        model, did_compile = bc.try_compile(model, mode="reduce-overhead", label="INT4 Inference (NF4)")
        compile_tag = " [compiled]" if did_compile else ""
        for seq_len in args.seq_lens:
            print(f"\n  INT4 Inference (NF4){compile_tag}  seq={seq_len}:")
            r = bc.sweep_batch_size(model, "INT4 Inference (NF4)", seq_len, n_gpus_model,
                                    vocab_size, compiled=did_compile,
                                    warmup=args.warmup, repeats=args.repeats)
            if r:
                results.append(r)
        del model; bc.flush()

    # ── INT8 Inference ──
    if not args.skip_int8:
        model = bc.load_model_int8(args.model)
        n_gpus_model = bc.count_gpus_used(model)
        print(f"\n  Parameters: {bc.count_parameters(model) / 1e9:.2f}B  |  GPUs used: {n_gpus_model}")
        model, did_compile = bc.try_compile(model, mode="reduce-overhead", label="INT8 Inference")
        compile_tag = " [compiled]" if did_compile else ""
        for seq_len in args.seq_lens:
            print(f"\n  INT8 Inference{compile_tag}  seq={seq_len}:")
            r = bc.sweep_batch_size(model, "INT8 Inference", seq_len, n_gpus_model,
                                    vocab_size, compiled=did_compile,
                                    warmup=args.warmup, repeats=args.repeats)
            if r:
                results.append(r)
        del model; bc.flush()

    # ── BF16 Training ──
    if not args.skip_train:
        model = bc.load_model_bf16(args.model)
        n_gpus_model = bc.count_gpus_used(model)
        print(f"\n  BF16 Training:")
        for seq_len in args.seq_lens:
            r = bc.bench_training(model, make_input(args.batch_size, seq_len), "BF16 Training",
                                  n_gpus_model, warmup=args.warmup, repeats=args.repeats)
            results.append(r)
            compiled_tag = " [compiled]" if r.compiled else ""
            print(f"    seq={seq_len}{compiled_tag}:  {r.ms_per_token:7.2f} ms/tok  |  "
                  f"{r.tok_per_sec_per_gpu:8.1f} tok/s/GPU")
        del model; bc.flush()

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"\n  {'Label':<28} {'Comp':>5} {'SeqLen':>6} {'ms/tok':>10} {'tok/s/GPU':>12}")
    print("  " + "─" * 65)
    for r in results:
        c = "yes" if r.compiled else "no"
        print(f"  {r.label:<28} {c:>5} {r.seq_len:>6} "
              f"{r.ms_per_token:>10.2f} {r.tok_per_sec_per_gpu:>12.1f}")

    # ── Ratios ──
    print("\n" + "=" * 72)
    print("  COMPUTE RATIOS  (Training wall time / Inference wall time)")
    print("=" * 72)

    for seq_len in args.seq_lens:
        sl = [r for r in results if r.seq_len == seq_len]
        train_r = next((r for r in sl if "Training" in r.label), None)
        if train_r is None:
            continue

        print(f"\n  Seq len = {seq_len}:")
        for r in sl:
            if "Inference" in r.label:
                ratio = (train_r.ms_per_token / r.ms_per_token
                         if r.ms_per_token > 0 else float("inf"))
                print(f"    {train_r.label} / {r.label}:  {ratio:.1f}x")

        print(f"    Theoretical (FLOP-only): 3.0x  |  INT8+compile: ~6–8x  |  INT4+compile: ~10–14x")

    # ── Save ──
    out_path = "bench_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
