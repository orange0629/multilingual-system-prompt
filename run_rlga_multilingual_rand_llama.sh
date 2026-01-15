#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=llama
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus=8
#SBATCH --time=72:00:00
#SBATCH -A qdj_project_owned3
#SBATCH -p spgpu2

# The application(s) to execute along with its input arguments and options:
#python /home/leczhang/prompting/scripts/inference_vllm.py
module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate /scratch/wangluxy_root/wangluxy1/leczhang/envs/prompting
# pip install langdetect absl nltk immutabledict
#python3.11-anaconda/2024.02

# export LD_LIBRARY_PATH=/home/leczhang/.local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
python scripts/multilingual_sprig/advanced_rlga_20250512_rand.py \
--model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--reward_path "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llama" \
--cache_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llamacache_rand" \
--output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llamacache_rand/results" \
--retrain