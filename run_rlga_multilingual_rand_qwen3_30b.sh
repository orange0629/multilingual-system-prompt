#!/bin/bash
#SBATCH --job-name=rlga_qwen3_30b
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus=8
#SBATCH --time=72:00:00
#SBATCH -A qdj_project_owned3
#SBATCH -p spgpu2

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate /scratch/wangluxy_root/wangluxy1/leczhang/envs/prompting

export HF_HOME="/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/models"

# GPU 0 → reward model; GPUs 1-6 → 3 workers × 2 GPUs each (tensor_parallel_size=2); GPU 7 unused
python scripts/multilingual_sprig/advanced_rlga_20250512_rand.py \
--model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
--reward_path "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/qwen3_30b" \
--cache_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/qwen3_30b_cache_rand" \
--output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/qwen3_30b_cache_rand/results" \
--gpus_per_model 2 \
--retrain
