import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
import os
import re
import flashinfer
print(flashinfer.__version__)


os.environ["VLLM_DISABLE_LOG_STATS"] = "1"

meta_cols = [
    "pred_label", "ground_truth", "is_correct", "model" ,
    "question_lang", "question_id", "benchmark", "output_len"
]


def strip_all_lines(text: str) -> str:
    return "\n".join([line.strip() for line in text.split("\n")])

def get_rtype2def_str(rtype2def: dict[str, str]) -> str:
    rtype2def["Others"] = (
        "This reasoning step is the continuation of the previous reasoning step, "
        "or it does not fall into any of the above categories."
    )
    return "\n".join(
        [f"{idx + 1}. {name}: {definition}" for idx, (name, definition) in enumerate(rtype2def.items())]
    )

def format_reasoning_steps(reasoning_steps: list[str]) -> str:
    return '\n'.join([f"<step_{i}>{s}</step_{i}>" for i, s in enumerate(reasoning_steps, 1)])

def get_classification_prompt(problem: str, reasoning: str, rtype2def_str: str) -> str:
    prompt = f"""
    Here is a problem and the reasoning process that an LLM generated when it tries to solve the problem.

    Problem: (enclosed in double backticks)
    ``
    {problem}
    ``

    Reasoning process: (enclosed in triple backticks, the reasoning process has been split into distinct reasoning steps in the format of <step_idx><reasoning_step_content></step_idx>)
    ```
    {reasoning}
    ```

    Your task is to classify each reasoning step into one of the following reasoning types: (specified by <type_index>. <type_name>: <definition>)
    {rtype2def_str}

    Generate the rationale before you make the classification.
    Provide your output in the following format:

    [Reasoning]
    <step_1><rationale_1><type_name_1></step_1>
    <step_2><rationale_2><type_name_2></step_2>
    ...

    [Final answer]
    <step_1><type_name_1></step_1>
    <step_2><type_name_2></step_2>
    ...
    """.strip()
    return strip_all_lines(prompt)

def wrap_qwen_prompt(user_content: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def extract_question(text: str) -> str:
    match = re.search(r"<\|im_start\|>user(.*?)<\|im_end\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def infer_tensor_parallel_size(tp_flag: int) -> int:
    if tp_flag and tp_flag > 0:
        return tp_flag
    env = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if env.strip():
        gpus = [x for x in env.split(",") if x.strip() != ""]
        if len(gpus) > 0:
            return len(gpus)
    try:
        import torch
        c = torch.cuda.device_count()
        return c if c > 0 else 1
    except Exception:
        return 1

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--reasoning_kind_def_json", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    return parser.parse_args()

def main():
    args = parse_args()
    tp_size = infer_tensor_parallel_size(args.tensor_parallel_size)

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
    )

    rtype2def = json.loads(args.reasoning_kind_def_json.read_text())
    rtype2def_str = get_rtype2def_str(rtype2def)

    df = pd.read_json(args.input_jsonl, lines=True)
    df["reasoning"] = df["formatted_steps"].apply(format_reasoning_steps)
    df["question"] = df["input"].apply(extract_question)

    df["prompt"] = df.apply(
        lambda row: wrap_qwen_prompt(
            get_classification_prompt(
                problem=row["question"].split("\n<|im_start|>user\n", 1)[-1].strip(),
                reasoning=row["reasoning"],
                rtype2def_str=rtype2def_str,
            )
        ),
        axis=1
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(df), args.batch_size), desc="Batch Inference"):
            batch_df = df.iloc[i:i + args.batch_size]
            try:
                outputs = llm.generate(batch_df["prompt"].tolist(), sampling_params)
            except Exception as e:
                print(f"[Batch Error] Skipping batch {i}-{i+len(batch_df)} due to: {e}")
                continue

            for row, out in zip(batch_df.itertuples(), outputs):
                text = out.outputs[0].text.strip()
                result = {"input": row.prompt, "output": text}
                for col in meta_cols:
                    result[col] = getattr(row, col)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
