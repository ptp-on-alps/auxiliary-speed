# Qwen2.5-72B Inference Benchmark — Alps (GH200)

Inference throughput for Qwen2.5-72B quantized to INT4 (bitsandbytes NF4) and INT8 on a single NVIDIA GH200 to compute auxiliaries for PTP training.

Task fits on single GPU in both quantizations. Throughput scales linearly with 100% efficiency since no coordination between tasks is necessary.

Metric: tokens/s/GPU across sequence lengths.



## Results

### INT4 (bitsandbytes NF4)

Quality: **P(same token | quantile coupling) = 0.9115** vs BF16 reference.

| Seq len | Batch size | ms/tok | tok/s/GPU |
|--------:|-----------:|-------:|----------:|
| 512 | 128 | 0.29 | **3,440** |
| 1024 | 128 | 0.29 | **3,414** |
| 2048 | 16 | 0.31 | **3,259** |
| 4096 | 16 | 0.31 | **3,264** |
| 8192 | 8 | 0.32 | **3,098** |

### INT8 (bitsandbytes)

Quality: **P(same token | quantile coupling) = 0.9364** vs BF16 reference.

| Seq len | Batch size | ms/tok | tok/s/GPU |
|--------:|-----------:|-------:|----------:|
| 512 | 32 | 0.38 | **2,608** |
| 1024 | 16 | 0.39 | **2,595** |
| 2048 | 8 | 0.39 | **2,594** |
| 4096 | 4 | 0.39 | **2,555** |
| 8192 | 2 | 0.41 | **2,454** |

> Methodology: `torch.compile` disabled, no Flash Attention 2, median over 10 timed iterations after 3 warmup steps. Largest batch size that fits in 94.5 GB HBM. Measured 2026-03-30.


## Specifications

### Hardware & Software

| Component | Specification |
|-----------|---------------|
| GPU | 1× NVIDIA GH200 120 GB (Hopper arch, SM 9.0, 94.5 GB usable HBM3e) |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 |
| CUDA | 12.9 |


### Model

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-72B (72.71B parameters) |
| INT4 format | bitsandbytes NF4, double quantization |
| INT8 format | bitsandbytes LLM.int8() |


### Launch Command

```bash
CUDA_VISIBLE_DEVICES=0 python inference_speed.py --model Qwen/Qwen2.5-72B \
    --no-compile --skip-train --warmup 1 --repeats 2
```
