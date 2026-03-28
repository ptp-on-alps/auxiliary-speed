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

echo "Quality check"
echo "====================="
# python quality_check.py --model Qwen/Qwen2.5-72B
echo "====================="
echo
echo
echo
echo "Inference speed"
echo "====================="
python inference_speed.py --model Qwen/Qwen2.5-72B --no-compile --skip-train
echo "====================="
