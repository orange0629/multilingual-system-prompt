# Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages

[![arXiv](https://img.shields.io/badge/arXiv-2512.02841-b31b1b.svg)](https://arxiv.org/abs/2512.02841)

**Authors:** Lechen Zhang, Yusheng Zhou, Tolga Ergen, Lajanugen Logeswaran, Moontae Lee, David Jurgens

## Abstract

System prompts provide a lightweight yet powerful mechanism for conditioning large language models (LLMs) at inference time. While prior work has focused on English-only settings, real-world deployments benefit from having a single prompt to operate reliably across languages. This paper presents a comprehensive study of how different system prompts steer models toward accurate and robust cross-lingual behavior. We propose a unified four-dimensional evaluation framework to assess system prompts in multilingual environments. Through large-scale experiments on five languages, three LLMs, and three benchmarks, we uncover that certain prompt components, such as CoT, emotion, and scenario, correlate with robust multilingual behavior. We develop a prompt optimization framework for multilingual settings and show it can automatically discover prompts that improve all metrics by 5-10%. Finally, we analyze over 10 million reasoning units and find that more performant system prompts induce more structured and consistent reasoning patterns, while reducing unnecessary language-switching. Together, we highlight system prompt optimization as a scalable path to accurate and robust multilingual LLM behavior.

## 🚀 Features

- **Three-part pipeline**: prompt synthesis → multilingual evaluation (Experiment 1) → prompt optimization (Experiment 2)
- **Multilingual Evaluation**: Evaluate system prompts across multiple languages and benchmarks
- **Prompt Optimization (SPRIG/RLGA)**: Automatic prompt discovery guided by a learned reward model
- **Reward Modeling**: Multi-metric reward model training for optimization guidance
- **Benchmark Support**: Math500, MMLU-Pro, and UniMoral

## 📋 Table of Contents

- [Installation](#installation)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Prompt Synthesis (Part 1)](#prompt-synthesis-part-1)
  - [Multilingual Evaluation (Experiment 1)](#multilingual-evaluation-experiment-1)
  - [Prompt Optimization (Experiment 2)](#prompt-optimization-experiment-2)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for model inference)
- Access to LLM models (Llama, Mistral, Qwen, Gemma)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd prompting_github
```

2. Install dependencies:
```bash
pip install torch transformers vllm datasets pandas numpy tqdm wandb openai filelock scipy
```

3. Configure model paths and API keys:
   - Update model cache directory in `scripts/lib/modelloader.py`
   - Set OpenAI API key in `scripts/prompt_generation/generate_prompt_components_v2.py` (if using prompt generation)
   - Set up Google translate key in `scripts//multilingual_sprig/translate.py`

## 🔁 Workflow

This codebase is organized around three parts:

1. **Prompt synthesis**: generate prompt components and assemble prompts.
2. **Experiment 1**: multilingual evaluation of prompts.
3. **Experiment 2**: prompt optimization with two folds:
   - **Train reward model**
   - **Run SPRIG optimizer (RLGA)**


## 📁 Project Structure

```
prompting_github/
├── data/
│   ├── benchmark/              # Benchmark datasets
│   └── task_prompts/          # Task-specific prompts by language
├── scripts/
│   ├── lib/                   # Core utilities
│   │   ├── dataloader.py      # Benchmark data loading
│   │   ├── modelloader.py     # Model loading and inference
│   │   └── eval/              # Evaluation metrics and utilities
│   ├── multilingual_sprig/    # Multilingual evaluation + SPRIG optimizer
│   │   ├── multilingual_eval_saveoutput_v3.py  # Main evaluation script
│   │   ├── advanced_rlga_20250512_rand.py       # RLGA optimization
│   │   ├── translate.py       # Prompt translation utilities
│   │   └── translate_prompt_components.py
│   ├── prompt_generation/     # Prompt component generation
│   │   ├── generate_prompt_components_v2.py
│   │   └── generate_prompts_from_components.py
│   └── reward_modeling/       # Reward model training
│       └── main_multi_metric.py
├── run_rlga_multilingual_rand_*.sh  # SPRIG optimizer launchers
└── scripts/reward_modeling/*.sh     # Reward model training launchers
└── README.md
```

## 📖 Usage

### Prompt Synthesis (Part 1)

Generate prompt components:

```bash
python scripts/prompt_generation/generate_prompt_components_v2.py
```

Assemble prompts from components:

```bash
python scripts/prompt_generation/generate_prompts_from_components.py
```

### Multilingual Evaluation (Experiment 1)
Example:

```bash
python scripts/multilingual_sprig/multilingual_eval_saveoutput_v3.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sys_lang "en" \
    --task_lang "zh" \
    --benchmark "unimoral" \
    --gpu_ids 0 1 \
    --output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/prompting"
```

Core CLI parameters used in the scripts:
- `--model_name`: Model identifier (e.g., `google/gemma-3-12b-it`, `Qwen/Qwen2.5-7B-Instruct`)
- `--sys_lang`: Language of the system prompt (`en`, `zh`, `es`, `fr`, `hi`)
- `--task_lang`: Language of the task/question
- `--benchmark`: Benchmark to evaluate (`math500`, `mmlu_pro`, `unimoral`)
- `--gpu_ids`: GPU IDs used for inference
- `--output_dir`: Output directory for results

### Prompt Optimization (Experiment 2)

Experiment 2 has two folds:

1. **Train reward model**  
   Example from `scripts/reward_modeling/train_reward_xlm_roberta_llama.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reward_modeling/main_multi_metric.py \
    --mode train \
    --model_checkpoint "FacebookAI/xlm-roberta-base" \
    --train_path "experiment2_rm_merged.jsonl" \
    --col_name "Llama-3.1-8B-Instruct" \
    --output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llama" \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --max_length 512
```

2. **Run SPRIG optimizer (RLGA)**  
   Example from `run_rlga_multilingual_rand_llama.sh`:

```bash
python scripts/multilingual_sprig/advanced_rlga_20250512_rand.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --reward_path "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llama" \
    --cache_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llamacache_rand" \
    --output_dir "/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/reward_models_multilingual/llamacache_rand/results" \
    --retrain
```

### Reasoning Pattern Analysis (Experiment 3)

Experiment 3 performs fine-grained reasoning pattern analysis on model outputs.  
It consists of two stages:

1. **Reasoning Step Segmentation**: split raw model outputs into atomic reasoning steps.
2. **Reasoning Type Classification**: classify each reasoning step into predefined reasoning categories using an LLM.

This experiment enables large-scale analysis of reasoning structure, step distribution, and reasoning-type frequency across prompts, languages, and benchmarks.

#### Step 1: Reasoning Step Segmentation

A trained token classification model is used to predict step boundaries in model-generated reasoning traces.  
Line breaks are replaced with a special separator token (`[SEP]`), and the model predicts whether each separator corresponds to a step boundary.

```bash
python scripts/response_analysis/segmentation/predict_step_splits.py \
    --model_path <SEGMENTATION_MODEL_PATH> \
    --input_jsonl <INPUT_JSONL> \
    --output_jsonl <OUTPUT_JSONL> \
    --batch_size 300
```

The output JSONL file contains:

- `formatted_steps`: list of segmented reasoning steps  
- `num_steps`: number of reasoning steps  
- original metadata fields preserved  

#### Step 2: Reasoning Type Classification

Each segmented reasoning step is classified into a predefined reasoning type using an instruction-following LLM (e.g., Qwen or LLaMA) via vLLM.

```bash
python scripts/response_analysis/classification/classify_reasoning_steps.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --input_jsonl <SEGMENTED_JSONL> \
    --reasoning_kind_def_json <REASONING_TYPE_DEFINITIONS_JSON> \
    --output_jsonl <OUTPUT_JSONL> \
    --batch_size 300
```


## 🔬 Evaluation Framework

Our four-dimensional evaluation framework assesses:

1. **Accuracy**: Performance across languages and tasks
2. **Robustness**: Consistency across language pairs
3. **Efficiency**: Output token variance and response length
4. **Consistency**: Cross-lingual reasoning pattern alignment

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{zhang2025crosslingualpromptsteerabilityaccurate,
      title={Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages}, 
      author={Lechen Zhang and Yusheng Zhou and Tolga Ergen and Lajanugen Logeswaran and Moontae Lee and David Jurgens},
      year={2025},
      eprint={2512.02841},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.02841}, 
}
```


## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---


