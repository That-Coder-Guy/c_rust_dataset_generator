#!/bin/bash
#SBATCH --job-name=c_rust_gen
#SBATCH --partition=killable.q
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=56:00:00
#SBATCH --output=/homes/hglenn2/c_rust_dataset/logs/%j.out
#SBATCH --error=/homes/hglenn2/c_rust_dataset/logs/%j.err

# -- Environment --------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c_rust_dataset
source ~/.cargo/env
module load CUDA/12.1.1

# -- Paths --------------------------------------------------------------------
export DATASET_OUTPUT_DIR=/homes/hglenn2/c_rust_dataset/output
export LLM_MODEL_PATH=/homes/hglenn2/c_rust_dataset/models/qwen3-coder-30b.gguf
export EMBED_MODEL_PATH=/homes/hglenn2/c_rust_dataset/models/nomic-embed-text.gguf
export CODE_EMBED_MODEL_PATH=/homes/hglenn2/c_rust_dataset/models/Qwen3-Embedding-8B-Q4_K_M.gguf

# -- Run ----------------------------------------------------------------------
cd /homes/hglenn2/c_rust_dataset

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python main.py --resume

echo "Job finished: $(date)"