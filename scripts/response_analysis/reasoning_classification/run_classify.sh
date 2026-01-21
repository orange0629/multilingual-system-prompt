#!/bin/bash                                                                                                                    
#SBATCH --job-name=classify         # Job name     
#SBATCH --mem=100G
#SBATCH --output=gemma_part1-%j.out             # Standard output and error log (%j expands to job ID)                                   
#SBATCH --partition=gpu                   # Partition name (as seen in sinfo)                                                                                                                                                                                       
#SBATCH --time=150:00:00                   

#SBATCH --gres=gpu:A6000:2                    

PYTHON=/opt/anaconda/bin/python3.12

MODEL_PATH="/shared/4/models/models--Qwen--Qwen3-14B"
INPUT_JSONL="sprig_outputs.jsonl"
REASONING_DEF_JSON="definition_behaviors.json"
OUTPUT_JSONL="sprig_outputs_classified.jsonl"
BATCH_SIZE=300

echo "=== SLURM ENV CHECK ==="
which python
python --version
python -c "import sys; print(sys.executable)"
python -c "import site; print(site.getsitepackages())"
python -c "import sys; print(sys.path)"
echo "======================="


$PYTHON classify_reasoning.py \
  --model_name "$MODEL_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --reasoning_kind_def_json "$REASONING_DEF_JSON" \
  --output_jsonl "$OUTPUT_JSONL" \
  --batch_size "$BATCH_SIZE"
