#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=reward_modeling
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH -A qdj_project_owned3
#SBATCH -p spgpu2

# The application(s) to execute along with its input arguments and options:
#python /home/leczhang/prompting/scripts/inference_vllm.py
module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate /scratch/wangluxy_root/wangluxy1/leczhang/envs/prompting

CUDA_VISIBLE_DEVICES=0 python main_multi_metric.py \
--mode train \
--model_checkpoint "FacebookAI/xlm-roberta-base" \
--train_path "experiment2_rm_merged.jsonl" \
--col_name "Gemma-3-12B-Instruct" \
--output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/gemma" \
--train_batch_size 16 \
--eval_batch_size 16 \
--max_length 512 \

CUDA_VISIBLE_DEVICES=0 python main_multi_metric.py \
--mode test \
--model_checkpoint "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/gemma" \
--train_path "experiment2_rm_merged.jsonl" \
--col_name "Gemma-3-12B-Instruct" \
--train_batch_size 16 \
--eval_batch_size 1 \
--max_length 512 \

# CUDA_VISIBLE_DEVICES=2,3 accelerate launch llava_model.py --mode train --model_checkpoint "llava-hf/llava-1.5-7b-hf" --train_path "/shared/0/datasets/mastodon/multimodal-data/oct-17-24-jsonl/two-classes/train.jsonl" --dev_path "/shared/0/datasets/mastodon/multimodal-data/oct-17-24-jsonl/two-classes/dev.jsonl" --output_dir "/shared/3/projects/national-culture/cache/imagemodels/LLaVa/oct-7/two-classes/fine_tuned_llava_model"