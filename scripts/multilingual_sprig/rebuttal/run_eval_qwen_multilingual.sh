#!/bin/bash
#SBATCH --job-name=run_evaluation
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4

# The application(s) to execute along with its input arguments and options:
#python /home/leczhang/prompting/scripts/inference_vllm.py
# module load cuda/12.6.3
# module load gcc/11.2.0
# source ~/.bashrc
# conda activate /scratch/wangluxy_root/wangluxy1/leczhang/envs/prompting
#python3.11-anaconda/2024.02
module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate /scratch/wangluxy_root/wangluxy1/leczhang/envs/prompting

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export HF_HUB_CACHE=/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/models


python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "sw" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "te" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "th" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"

python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "sw" --benchmark "math500" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "te" --benchmark "math500" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "th" --benchmark "math500" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"

python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "sw" --benchmark "unimoral" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "te" --benchmark "unimoral" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen2.5-7B-Instruct" --sys_lang "en" --task_lang "th" --benchmark "unimoral" --gpu_ids 0 1 2 3 4 5 6 7 --gpus_per_model 2 --output_dir "./results/"
