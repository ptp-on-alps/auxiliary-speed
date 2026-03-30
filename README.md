# Auxiliary speed

Benchmarks inference throughput and quantization quality for large language models on CSCS Alps (NVIDIA GH200). The primary use case is measuring the throughput at computing the auxiliaries for the PTP objective.

## Scripts

| Script | Purpose |
|--------|---------|
| `inference_speed.py` | Measure tokens/s/GPU across sequence lengths and batch sizes |
| `quality_check.py` | Measure P(same token | quantile coupling) vs. a BF16 reference |
| `run.sh` | SLURM job script — runs quality check then inference benchmark |


Result summary to be found in [Qwen2.5-72B-Inference-alps.md](Qwen2.5-72B-Inference-alps.md).


Details about the runs in the logs:

- [logs/3039189.out](logs/3039189.out) for script output.
- [logs/3039189.err](logs/3039189.err) for the loading details.

