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
export HF_HUB_CACHE=/shared/4/models/


# export LD_LIBRARY_PATH=/home/leczhang/.local/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "en" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "zh" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "es" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "fr" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "hi" --benchmark "mmlu_pro" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"

# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "en" --benchmark "math500" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "zh" --benchmark "math500" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "es" --benchmark "math500" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "fr" --benchmark "math500" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "hi" --benchmark "math500" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"

# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "en" --benchmark "unimoral" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "zh" --benchmark "unimoral" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "es" --benchmark "unimoral" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "fr" --benchmark "unimoral" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
# python multilingual_eval_saveoutput_v3.py --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" --sys_lang "en" --task_lang "hi" --benchmark "unimoral" --gpu_ids 0 1 2 3 --gpus_per_model 2 --output_dir "./results/"
