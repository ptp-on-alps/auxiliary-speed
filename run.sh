#!/bin/bash
#SBATCH --job-name=inference-speed
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --environment=/capstor/scratch/cscs/rgeyer/megatron-container/megatron.toml
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fdraxler/inference-speed/logs/%j.out
#SBATCH --error=/capstor/scratch/cscs/fdraxler/inference-speed/logs/%j.err

cd /capstor/scratch/cscs/rgeyer/ptp-on-alps/Megatron-LM/
. .venv/bin/activate
cd /capstor/scratch/cscs/fdraxler/inference-speed/

pip install compressed-tensors

# We planned to run with Apertus-70B-2509, but the cluster was down when we submitted the job
# MODEL=swiss-ai/Apertus-70B-2509
MODEL=Qwen/Qwen2.5-72B

echo "Quantize model"
echo "====================="
python quantize.py --model $MODEL --bits 4
python quantize.py --model $MODEL --bits 8
echo "====================="
echo
echo
echo
echo "Quality check"
echo "====================="
python quality_check.py --model $MODEL
echo "====================="
echo
echo
echo
echo "Inference speed (single GPU)"
echo "====================="
CUDA_VISIBLE_DEVICES=0 python inference_speed.py --model $MODEL --no-compile --skip-train --warmup 1 --repeats 2
echo "====================="
