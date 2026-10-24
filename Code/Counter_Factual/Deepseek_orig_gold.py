#!/usr/bin/env python3
"""
Orig_gold Judge (DeepSeek via OpenRouter) 

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
# ★ 路径配置
# =====================================================
BASE_DIR   = Path(r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Counter_Factual")
ORIG_DIR   = BASE_DIR / "orig_gold_model_answer"
OUTPUT_DIR = BASE_DIR / "orign_gold_judge"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    {"model": "Qwen3-14B", "path": ORIG_DIR / "orign_gold_context_gen_Qwen3-14B.jsonl"},
    {"model": "Qwen3-32B", "path": ORIG_DIR / "orign_gold_context_gen_Qwen3-32B.jsonl"},
    {"model": "DeepSeek-V4.1-Flash", "path": ORIG_DIR / "orign_gold_context_gen_DeepSeek-V4.1-Flash.jsonl"},
    {"model": "Qwen3.8-27B", "path": ORIG_DIR / "orign_gold_context_gen_Qwen3.8-27B.jsonl"},
]

JUDGE_CONFIG = {
    "DeepSeek-V4.1-Flash": {
        "model_name":      "deepseek/deepseek-v4.1-flash",
        "base_url":        "https://openrouter.ai/api/v1",
        "api_key":         " ",
        "max_workers":     32,
        "enable_thinking": True,
        "temperature":     0.0,
        "max_tokens":      8192,
        "backend":         "openrouter",
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

# =====================================================
# ★ 字段读取
# =====================================================
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

# =====================================================
# ★ 解析 judge 输出
# =====================================================
def parse_judge_output(raw_text):
    if not raw_text:
        return "", "incorrect", True
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip().replace("</think>", "").strip()
    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)", text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text
    m = re.search(r"Final Judgment:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        return explanation, m.group(1).lower(), False
    return explanation, "incorrect", True

# =====================================================
# ★ 单次 API 调用
# =====================================================
def call_judge_once(client, cfg, prompt):
    kwargs = {
        "model": cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8192,
        "timeout": 600,
    }
    if cfg.get("backend") == "openrouter" and cfg.get("enable_thinking"):
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = getattr(msg, "content", "") or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", "") or ""
    if not reasoning:
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content") or ""
    return (
        f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content,
        response.choices[0].finish_reason,
    )

# =====================================================
# ★ 单条处理（带重试）
# =====================================================
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
            print(f"API报错 (重试 {attempt}/{MAX_RETRY}): {e}")
            time.sleep(2 ** attempt)

    new_item = item.copy()
    new_item.update({
        "judge_raw_output": "ERROR",
        "judge_explanation": "ERROR",
        "judge_result": "incorrect",
    })
    return idx, new_item

# =====================================================
# ★ 单轮
# =====================================================
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

# =====================================================
# ★ 合并 3 轮
# =====================================================
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

# =====================================================
# ★ 主程序
# =====================================================
def main():
    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"\n🚀 Judge: {judge_name}")

        for f in INPUT_FILES:
            if not f["path"].exists():
                print(f"⚠️ 找不到文件，跳过: {f['path']}")
                continue

            with open(f["path"], encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]

            print(f"\n📁 成功读取文件: {f['path'].name} ({len(samples)} 条)")
            model_token = f["model"]

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = OUTPUT_DIR / f"orign_gold_{model_token}_{judge_name}_run{run}.jsonl"
                run_files.append(out_path)

                if out_path.exists():
                    with open(out_path, encoding="utf-8") as fp:
                        if sum(1 for line in fp if line.strip()) == len(samples):
                            print(f"  ⏭️ run{run} 已存在，跳过")
                            continue

                results = run_one_round(
                    samples, client, cfg,
                    desc=f"  {model_token} run{run}/{NUM_RUNS}",
                )
                with open(out_path, "w", encoding="utf-8") as fp:
                    for item in results:
                        fp.write(json.dumps(item, ensure_ascii=False) + "\n")

            final_judged_path = OUTPUT_DIR / f"orign_gold_{model_token}_{judge_name}_judged.jsonl"
            merge_runs(run_files, final_judged_path)

if __name__ == "__main__":
    main()