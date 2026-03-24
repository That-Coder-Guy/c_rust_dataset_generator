#!/bin/bash
#SBATCH --job-name=c_rust_gen
#SBATCH --partition=killable.q
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=56:00:00
#SBATCH --output=/homes/hglenn2/c_rust_dataset/logs/%j.out
#SBATCH --error=/homes/hglenn2/c_rust_dataset/logs/%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c_rust_dataset
module load CUDA/12.1.1

# ── Paths ─────────────────────────────────────────────────────────────────────
export DATASET_OUTPUT_DIR=/homes/hglenn2/c_rust_dataset/output
export LLM_MODEL_PATH=/homes/hglenn2/c_rust_dataset/models/qwen2.5-coder-32b.gguf
export EMBED_MODEL_PATH=/homes/hglenn2/c_rust_dataset/models/nomic-embed-text.gguf

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p /homes/hglenn2/c_rust_dataset/logs
cd /homes/hglenn2/c_rust_dataset

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python main.py --resume

echo "Job finished: $(date)"
