#!/usr/bin/env python3
"""
Benchmark inference speed across one or more models.

Usage:
  python inference_speed.py --models swiss-ai/Apertus-70B-2509 \
                                     RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16 \
                             --no-compile --skip-train
"""

import argparse
import json
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

import bench_common as bc


def main():
    parser = argparse.ArgumentParser(description="Inference and training speed benchmark")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="One or more HuggingFace model names or local paths")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Timed iterations (median reported)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training benchmark")
    args = parser.parse_args()

    if args.no_compile:
        bc._compile_disabled = True

    print("=" * 72)
    print("  Inference Speed Benchmark")
    print("  (compile + TF32 + fused optim)")
    print("=" * 72)

    n_gpus = torch.cuda.device_count()
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"GPUs:    {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  "
              f"({props.total_memory / (1024**3):.1f} GB, "
              f"SM {props.major}.{props.minor})")
    print(f"\n  TF32 matmul:   {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  torch.compile: {not bc._compile_disabled}")

    all_results = []

    for model_name in args.models:
        print(f"\n{'=' * 72}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 72}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size

        def make_input(bs, seq_len):
            return torch.randint(100, vocab_size - 100, (bs, seq_len), device="cuda")

        # ── Inference ──
        model = bc.load_model(model_name)
        n_gpus_model = bc.count_gpus_used(model)
        print(f"\n  Parameters: {bc.count_parameters(model) / 1e9:.2f}B  |  GPUs used: {n_gpus_model}")
        model, did_compile = bc.try_compile(model, mode="reduce-overhead", label="Inference")
        compile_tag = " [compiled]" if did_compile else ""
        for seq_len in args.seq_lens:
            print(f"\n  Inference{compile_tag}  seq={seq_len}:")
            r = bc.sweep_batch_size(model, model_name, seq_len, n_gpus_model,
                                    vocab_size, compiled=did_compile,
                                    warmup=args.warmup, repeats=args.repeats)
            if r:
                all_results.append(r)
        del model; bc.flush()

        # ── BF16 Training ──
        if not args.skip_train:
            model = bc.load_model_for_training(model_name)
            n_gpus_model = bc.count_gpus_used(model)
            print(f"\n  BF16 Training:")
            for seq_len in args.seq_lens:
                r = bc.bench_training(model, make_input(args.batch_size, seq_len),
                                      f"{model_name} [train]", n_gpus_model,
                                      warmup=args.warmup, repeats=args.repeats)
                all_results.append(r)
                compiled_tag = " [compiled]" if r.compiled else ""
                print(f"    seq={seq_len}{compiled_tag}:  {r.ms_per_token:7.2f} ms/tok  |  "
                      f"{r.tok_per_sec_per_gpu:8.1f} tok/s/GPU")
            del model; bc.flush()

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"\n  {'Label':<50} {'Comp':>5} {'SeqLen':>6} {'ms/tok':>10} {'tok/s/GPU':>12}")
    print("  " + "─" * 87)
    for r in all_results:
        c = "yes" if r.compiled else "no"
        print(f"  {r.label:<50} {c:>5} {r.seq_len:>6} "
              f"{r.ms_per_token:>10.2f} {r.tok_per_sec_per_gpu:>12.1f}")

    out_path = "bench_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
