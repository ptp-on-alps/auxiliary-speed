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

MODEL=swiss-ai/Apertus-70B-2509
# COMPRESSED_MODEL=RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16
# MODEL=Qwen/Qwen2.5-72B

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
echo "Inference speed"
echo "====================="
CUDA_VISIBLE_DEVICES=0 python inference_speed.py --model $MODEL --no-compile --skip-train --warmup 1 --repeats 2
echo "====================="
