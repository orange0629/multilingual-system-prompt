#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge result files from results/ into a single JSONL file with aggregated metrics,
matching the format of experiment2_rm_merged.jsonl.

Each result file is named:
  full_outputs_log_{model}_{benchmark}_{sys_lang}_{q_lang}.jsonl

We read all files, reconstruct the DataFrame expected by compute_metrics(), then
write the aggregated metrics as JSONL.
"""

import os
import re
import json
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results_merged.jsonl")


def is_all_equal(labels):
    return 1 if len(set(labels)) == 1 else 0


def compute_metrics(df):
    df_len_var_step1 = (
        df
        .groupby(["benchmark", "prompt", "model", "question_id"])
        .agg(output_len_var=("output_len", "var"))
        .reset_index()
    )

    df_len_var = (
        df_len_var_step1
        .groupby(["benchmark", "model", "prompt"])
        .agg(output_tokens_var=("output_len_var", "mean"))
        .reset_index()
    )

    df_acc_var_len = (
        df
        .groupby(["benchmark", "prompt", "question_lang", "model"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby(["benchmark", "model", "prompt"])
        .agg(
            acc_mean=("is_correct", "mean"),
            acc_var=("is_correct", "var"),
        )
        .reset_index()
    )

    df_consistency = (
        df
        .groupby(["benchmark", "model", "prompt", "question_id"])["pred_label"]
        .apply(is_all_equal)
        .reset_index(name="all_equal_consistency")
        .groupby(["benchmark", "model", "prompt"])["all_equal_consistency"]
        .mean()
        .reset_index(name="consistency")
    )

    df_merged = pd.merge(df_acc_var_len, df_consistency, on=["benchmark", "model", "prompt"], how="left")
    df_merged = pd.merge(df_merged, df_len_var, on=["benchmark", "model", "prompt"], how="left")
    df_merged = df_merged.groupby(["model", "prompt"]).mean(numeric_only=True).reset_index()
    return df_merged


def parse_filename(fname):
    """Extract (model, benchmark, sys_lang, q_lang) from filename."""
    # full_outputs_log_{model}_{benchmark}_{sys_lang}_{q_lang}.jsonl
    m = re.match(
        r"full_outputs_log_(.+?)_(math500|mmlu_pro|unimoral)_([a-z]+)_([a-z]+)\.jsonl",
        fname,
    )
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None


def load_all_results():
    all_rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"Skipping unrecognized file: {fname}")
            continue
        model, benchmark, sys_lang, q_lang = parsed
        fpath = os.path.join(RESULTS_DIR, fname)
        print(f"Loading {fname} ...")

        # Track per-(benchmark, prompt) question index for consistent question_id
        # across language files (same question order, just translated).
        prompt_q_counter = {}

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt = entry["system_prompt"]
                key = (benchmark, prompt)
                idx = prompt_q_counter.get(key, 0)
                prompt_q_counter[key] = idx + 1

                all_rows.append({
                    "benchmark": benchmark,
                    "model": model,
                    "prompt": prompt,
                    "question_lang": q_lang,
                    "question_id": f"{benchmark}_{idx}",
                    "output_len": len(entry.get("model_output", "") or ""),
                    "is_correct": int(entry.get("is_correct", 0)),
                    "pred_label": str(entry.get("pred_label", "")),
                })

    return pd.DataFrame(all_rows)


def main():
    df = load_all_results()
    print(f"Loaded {len(df)} rows total.")
    print("Computing metrics ...")
    metrics_df = compute_metrics(df)
    print(f"Computed metrics for {len(metrics_df)} (model, prompt) pairs.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in metrics_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
