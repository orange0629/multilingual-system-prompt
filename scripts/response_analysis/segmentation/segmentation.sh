#!/bin/bash                                                                                                                    
#SBATCH --job-name=segmentation        # Job name                                                                               
#SBATCH --output=segmetation_log-%j.out                    
#SBATCH --partition=gpu                   # Partition name (as seen in sinfo)                                                                                                                                                                                       
#SBATCH --time=10:00:00                   
#SBATCH --gres=gpu:A100:1

source ~/.bashrc

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASHINFER_COMPUTE_CAPS=80


BATCH_SIZE=25
input_file="inference_batch_1.txt"
MODEL_PATH="appier-ai-research/reasoning-segmentation-model-v0"

mkdir -p logs

while IFS= read -r filepath || [ -n "$filepath" ]; do
    if [ -f "$filepath" ]; then
        echo "Processing file: $filepath"
    else
        echo -e "\033[33mWarning: File '$filepath' does not exist.\033[0m"
        continue
    fi

    output_dir=$(dirname "$filepath")/results
    mkdir -p "$output_dir"

    filename=$(basename "$filepath")
    output_file="$output_dir/$filename"

    echo "Saving to $output_file"

    /opt/anaconda/bin/python segmentation.py \
        --model_path "$MODEL_PATH" \
        --input_jsonl "$filepath" \
        --output_jsonl "$output_file" \
        --batch_size "$BATCH_SIZE"
done < "$input_file"

echo "All files processed."