"""
Shared utilities, model loaders, and benchmark primitives.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.amp import autocast
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Global CUDA optimizations (set at import time, before any model loading)
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label: str
    seq_len: int
    tokens: int
    wall_sec: float
    n_gpus: int = 1
    compiled: bool = False
    ms_per_token: float = 0.0
    tok_per_sec_per_gpu: float = 0.0

    def __post_init__(self):
        if self.tokens > 0 and self.wall_sec > 0:
            self.ms_per_token = (self.wall_sec / self.tokens) * 1000
            self.tok_per_sec_per_gpu = (self.tokens / self.wall_sec) / max(self.n_gpus, 1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_gpus_used(model) -> int:
    devices = {p.device for p in model.parameters() if p.device.type == "cuda"}
    return max(len(devices), 1)


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@contextmanager
def cuda_timer():
    """Accurate GPU timing via CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    yield lambda: (start.elapsed_time(end) / 1000.0)  # seconds
    end.record()
    torch.cuda.synchronize()


# Reference text for quality comparison (real prose, not random tokens)
_QUALITY_TEXT = """\
The Transformer architecture was introduced in "Attention is All You Need" by Vaswani et al. in 2017.
It relies entirely on self-attention mechanisms to draw global dependencies between input and output,
dispensing with recurrence and convolutions entirely. Each layer has two sub-layers: a multi-head
self-attention mechanism and a position-wise fully connected feed-forward network. Residual connections
are employed around each sub-layer, followed by layer normalization. Large language models such as
GPT and Llama are decoder-only Transformer variants pretrained on vast text corpora using next-token
prediction. During inference the model autoregressively generates tokens by sampling from the predicted
distribution over the vocabulary. Quantization reduces the precision of model weights to lower bit
widths, trading some accuracy for significant reductions in memory footprint and computation cost.
Common schemes include 8-bit integers (INT8) and 4-bit formats such as NF4, which uses a non-uniform
quantization grid optimized for normally distributed weights. The trade-off between model quality and
inference efficiency is central to deploying large models in production settings where latency and
throughput matter. Techniques such as GPTQ, AWQ, and bitsandbytes allow practitioners to quantize
models post-training with minimal calibration data and acceptable degradation in downstream benchmarks.
"""

_compile_disabled = False


def try_compile(model, mode: str = "reduce-overhead", label: str = "") -> tuple:
    """Attempt torch.compile; fall back gracefully.

    Returns (possibly_compiled_model, did_compile_succeed).
    """
    if _compile_disabled:
        return model, False
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        print(f"    torch.compile({mode}) enabled for {label}")
        return compiled, True
    except Exception as e:
        print(f"    torch.compile failed for {label}, falling back: {e}")
        return model, False


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load a model as-is from HuggingFace (respects embedded quantization_config)."""
    print(f"\n  Loading {model_name} ...")
    n_gpus = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: "90GiB" for i in range(n_gpus)},
        cache_dir=".",
    )
    model.eval()
    return model


def load_model_for_training(model_name: str):
    print(f"\n  Loading {model_name} in BF16 for training ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=".",
    )
    model.train()
    # use_reentrant=False is required for torch.compile compatibility
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_inference(model, input_ids: torch.Tensor, label: str, n_gpus: int,
                    compiled: bool = False,
                    warmup: int = 5, repeats: int = 20) -> BenchResult:
    """Time a single forward pass (model must already be compiled if desired)."""
    seq_len = input_ids.shape[1]
    tokens = input_ids.numel()

    for _ in range(warmup):
        with torch.inference_mode():
            model(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        with cuda_timer() as get_elapsed:
            with torch.inference_mode():
                model(input_ids)
        times.append(get_elapsed())

    median_sec = sorted(times)[len(times) // 2]
    return BenchResult(label=label, seq_len=seq_len, tokens=tokens,
                       wall_sec=median_sec, n_gpus=n_gpus, compiled=compiled)


def sweep_batch_size(model, label: str, seq_len: int, n_gpus: int,
                     vocab_size: int, compiled: bool,
                     warmup: int = 3, repeats: int = 10,
                     start_bs: int = 256) -> BenchResult | None:
    """Halve batch size from start_bs until no OOM; return the largest working result."""
    bs = start_bs
    while bs >= 1:
        input_ids = torch.randint(100, vocab_size - 100, (bs, seq_len), device="cuda")
        try:
            r = bench_inference(model, input_ids, label, n_gpus,
                                compiled=compiled, warmup=warmup, repeats=repeats)
            print(f"      bs={bs:>4}:  {r.ms_per_token:7.2f} ms/tok  |  "
                  f"{r.tok_per_sec_per_gpu:8.1f} tok/s/GPU")
            return r
        except torch.cuda.OutOfMemoryError:
            flush()
            print(f"      bs={bs:>4}:  OOM — halving")
            bs //= 2
    print("      No working batch size found (bs=1 OOM)")
    return None


def bench_training(model, input_ids: torch.Tensor, label: str, n_gpus: int,
                   warmup: int = 5, repeats: int = 20) -> BenchResult:
    """Benchmark full training step with all speedups."""
    seq_len = input_ids.shape[1]
    tokens = input_ids.numel()

    # Fused AdamW — single kernel for the optimizer step (Hopper/Ampere)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
        print("    Fused AdamW enabled")
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
        print("    foreach AdamW fallback (fused not available)")

    # torch.compile — "max-autotune" for training (tries more kernel configs)
    model, compiled = try_compile(model, mode="max-autotune", label=label)

    # BF16 autocast — gives compiler more fusion opportunities
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=input_ids)
            loss = out.loss
        loss.backward()
        optimizer.step()

    # Extended warmup for compile tracing
    effective_warmup = warmup + (8 if compiled else 0)
    for _ in range(effective_warmup):
        train_step()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        with cuda_timer() as get_elapsed:
            train_step()
        times.append(get_elapsed())

    median_sec = sorted(times)[len(times) // 2]
    return BenchResult(label=label, seq_len=seq_len, tokens=tokens,
                       wall_sec=median_sec, n_gpus=n_gpus, compiled=compiled)
