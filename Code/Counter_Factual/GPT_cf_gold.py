#!/usr/bin/env python3
"""
CF-gold GPT-6-Luna Judge - 反事实 (Old Gold)
"""

import os
import re
import json
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 路径配置（旧/反事实，参照输入目录命名）
# =====================================================
BASE_DIR = Path(r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Counter_Factual")

INPUT_DIR  = BASE_DIR / "orign_context_gen"
OUTPUT_DIR = BASE_DIR / "orign_context_gen_judge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["DeepSeek-V4.1-Flash", "Qwen3-14B", "Qwen3-32B", "Qwen3.8-27B"]

# =====================================================
# ★ GPT-6-Luna (ChatAnywhere) 配置
# =====================================================
JUDGE_CONFIG = {
    "GPT-6-Luna": {
        "model_name":      "gpt-6-luna",
        "base_url":        "https://api.chatanywhere.tech/v1",
        "api_key":         "  ",
        "max_workers":     32,
        "max_tokens":      8192,
        "temperature":     None,
    },
}

NUM_RUNS, REQUEST_TIMEOUT, MAX_RETRY = 3, 600, 3

# =====================================================
# ★ Judge Prompt
# =====================================================
JUDGE_PROMPT = """Judge whether [model_answer], the response to [question], is correct based on [gold_answer], the reference answer.

[question]: {question}

[model_answer]: {model_answer}

[gold_answer]: {gold_answer}

Evaluate semantic equivalence between [model_answer] and [gold_answer] in the context of [question], rather than exact wording.
If [gold_answer] provides alternative answers, matching any one complete alternative is sufficient.
Judge [model_answer] as correct if it is semantically equivalent to [gold_answer]; otherwise, judge it as incorrect.

Return exactly two lines:
Explanation: <a brief explanation on a single line>
Final Judgment: <correct or incorrect>

The final judgment must be exactly correct or incorrect, in lowercase and without quotation marks. Do not include any additional text."""

def get_question(item): return item.get("question", "")
def get_gold(item):
    g = item.get("gold", "")
    return " / ".join(str(x) for x in g) if isinstance(g, list) else g
def get_predicted(item): return item.get("model_answer_orig", "") or ""

def build_prompt(item):
    return JUDGE_PROMPT.format(
        question=get_question(item),
        model_answer=get_predicted(item),
        gold_answer=get_gold(item),
    )

def parse_judge_output(raw_text):
    if not raw_text:
        return "", "incorrect", True
    text = raw_text
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)", text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    m = re.search(r"Final Judgment:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        return explanation, m.group(1).lower(), False
    return explanation, "incorrect", True

def call_judge_once(client, cfg, prompt):
    kwargs = {
        "model":    cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "timeout":  REQUEST_TIMEOUT,
    }
    if cfg.get("max_tokens") is not None:
        kwargs["max_completion_tokens"] = cfg["max_tokens"]
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        err = str(e)
        if "max_completion_tokens" in err and "not supported" in err:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = cfg["max_tokens"]
            response = client.chat.completions.create(**kwargs)
        elif "max_tokens" in err and "not supported" in err:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = cfg["max_tokens"]
            response = client.chat.completions.create(**kwargs)
        elif "temperature" in err and "not supported" in err:
            kwargs.pop("temperature", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    choice = response.choices[0]
    return choice.message.content or "", choice.finish_reason

def process_one(idx, item, client, cfg):
    prompt = build_prompt(item)
    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw, finish = call_judge_once(client, cfg, prompt)
            expl, result, err = parse_judge_output(raw)
            if finish == "length" or err:
                time.sleep(2)
                continue
            new_item = item.copy()
            new_item.update({
                "judge_raw_output": raw,
                "judge_explanation": expl,
                "judge_result": result,
            })
            return idx, new_item
        except Exception as e:
            time.sleep(min(2 ** attempt, 10))

    new_item = item.copy()
    new_item.update({
        "judge_raw_output": "ERROR",
        "judge_explanation": "ERROR",
        "judge_result": "incorrect",
    })
    return idx, new_item

def run_one_round(samples, client, cfg, desc):
    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {
            executor.submit(process_one, i, item, client, cfg): i
            for i, item in enumerate(samples)
        }
        with tqdm(total=len(samples), desc=desc) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)
    return results

def merge_runs(run_files, merged_path):
    runs = []
    for f in run_files:
        with open(f, encoding="utf-8") as fp:
            runs.append([json.loads(line) for line in fp if line.strip()])

    merged = []
    for i in range(len(runs[0])):
        base = runs[0][i].copy()
        for k in ["judge_raw_output", "judge_explanation", "judge_result"]:
            base.pop(k, None)
        res = [runs[k][i].get("judge_result", "incorrect") for k in range(3)]
        base["judge_answer"] = Counter(res).most_common(1)[0][0]
        base["judge_results"] = res
        base["judge_explanations_3"] = [runs[k][i].get("judge_explanation", "") for k in range(3)]
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ 合并完毕: {merged_path.name}")

def main():
    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"\n🚀 使用 Judge: {judge_name} (ChatAnywhere) - 反事实 Old Gold")
        print(f"   输入: {INPUT_DIR}")
        print(f"   输出: {OUTPUT_DIR}")

        for model in MODELS:
            file_name = f"orign_context_gen_{model}.jsonl"
            old_path = INPUT_DIR / file_name

            if not old_path.exists():
                print(f"\n⚠️ 找不到文件，跳过: {old_path}")
                continue

            with open(old_path, encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]

            print(f"\n📁 [Old] 成功读取: {file_name} ({len(samples)} 条)")
            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = OUTPUT_DIR / f"orign_context_gen_{model}_{judge_name}_run{run}.jsonl"
                run_files.append(out_path)

                if out_path.exists():
                    with open(out_path, encoding="utf-8") as fp:
                        if sum(1 for line in fp if line.strip()) == len(samples):
                            print(f"  ⏭️ run{run} 已存在，跳过")
                            continue

                results = run_one_round(samples, client, cfg, desc=f"  run{run}/{NUM_RUNS}")
                with open(out_path, "w", encoding="utf-8") as fp:
                    for item in results:
                        fp.write(json.dumps(item, ensure_ascii=False) + "\n")

            final_path = OUTPUT_DIR / f"orign_context_gen_{model}_{judge_name}_judged.jsonl"
            merge_runs(run_files, final_path)

if __name__ == "__main__":
    main()