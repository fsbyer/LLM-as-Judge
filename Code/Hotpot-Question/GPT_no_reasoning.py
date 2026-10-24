#!/usr/bin/env python3
"""
用 ChatAnywhere 的 gpt-6-luna 对 HQ 做 judge
vanilla prompt (Explanation + Final Judgment)，3 轮全存
"""

import os
import re
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# =====================================================
# ★ 配置区
# =====================================================

BASE = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge"

INPUT_DIR  = os.path.join(BASE, "Output_WithoutReasoning")
OUTPUT_DIR = os.path.join(BASE, "vanilla", "hq")

TARGET_FILES = [
    "hq_deepseek-v4.1-flash.jsonl",
    "hq_Qwen3-8B.jsonl",
    "hq_Qwen3-14B.jsonl",
    "hq_Qwen3-32B.jsonl",
    "hq_Qwen38_27b.jsonl",
]

JUDGE_CONFIG = {
    "GPT-6-Luna": {
        "model_name":      "gpt-6-luna",
        "base_url":        "https://api.chatanywhere.tech/v1",
        "api_key":         " ",
        "max_workers":     32,
        "max_tokens":      8192,
        "temperature":     None,
        "enable_thinking": False,
    },
}

NUM_RUNS    = 3
MAX_RETRIES = 3
RETRY_DELAY = 5

# =====================================================
# ★ Vanilla Prompt
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

def get_question(item):
    return item.get("question") or item.get("problem") or ""

def get_gold(item):
    g = item.get("gold") or item.get("answer") or item.get("ground_truth") or ""
    if isinstance(g, list):
        g = " / ".join(str(x) for x in g)
    return g

def get_predicted(item):
    if item.get("model_answer"):
        return item["model_answer"]
    if item.get("model_output"):
        return item["model_output"]
    raw = item.get("raw_output", "")
    if not raw:
        return ""
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

def build_prompt(item):
    return JUDGE_PROMPT.format(
        question=get_question(item),
        model_answer=get_predicted(item),
        gold_answer=get_gold(item),
    )

# =====================================================
# ★ 解析 judge 输出（Explanation + Final Judgment）
# =====================================================

def parse_judge_output(raw):
    if not raw:
        return "", "", "incorrect"

    text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = text.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    m = re.search(r"Final Judgment:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        result = m.group(1).lower()
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", text.lower())
        result = labels[-1] if labels else "incorrect"

    return text, explanation, result

# =====================================================
# ★ 构造 API 调用参数（兼容 max_completion_tokens / temperature）
# =====================================================

def build_kwargs(cfg, prompt):
    kwargs = {
        "model":    cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg.get("max_tokens") is not None:
        kwargs["max_completion_tokens"] = cfg["max_tokens"]
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("enable_thinking"):
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}
    return kwargs

# =====================================================
# ★ 单条调用（带重试 + 自动 fallback）
# =====================================================

def call_one(client, item, cfg):
    prompt = build_prompt(item)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = build_kwargs(cfg, prompt)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                if "max_completion_tokens" in err and "not supported" in err:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = cfg["max_tokens"]
                    response = client.chat.completions.create(**kwargs)
                elif "temperature" in err and "not supported" in err:
                    kwargs.pop("temperature", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""
            reasoning_field = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            raw_output = (f"<think>\n{reasoning_field}\n</think>\n\n{content}"
                          if reasoning_field else content)

            judge_raw, explanation, result = parse_judge_output(raw_output)
            return {
                "raw_output":  raw_output,
                "content":     content,
                "explanation": explanation,
                "result":      result,
            }
        except Exception as e:
            err_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    ⚠️ 重试 {attempt+1}/{MAX_RETRIES} ({wait}s): {err_msg[:120]}")
                time.sleep(wait)
            else:
                return {
                    "raw_output":  f"ERROR: {err_msg}",
                    "content":     "",
                    "explanation": f"ERROR: {err_msg}",
                    "result":      "ERROR",
                }

# =====================================================
# ★ 单轮
# =====================================================

def run_one_round(samples, client, cfg, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        r = call_one(client, item, cfg)
        return idx, r

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {executor.submit(process_one, i, item): i
                   for i, item in enumerate(samples)}
        with tqdm(total=total, desc=desc) as pbar:
            for future in as_completed(futures):
                idx, r = future.result()
                results[idx] = r
                pbar.update(1)

    return results

# =====================================================
# ★ 单个文件 3 轮 + 合并
# =====================================================

def process_one_file(input_path, output_dir, judge_name, cfg, client):
    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    print(f"\n📁 {base_name}  ({len(samples)} 条)")

    run_results = []
    for run in range(1, NUM_RUNS + 1):
        res = run_one_round(samples, client, cfg, desc=f"  run{run}/{NUM_RUNS}")
        run_results.append(res)

    final_rows = []
    for i, item in enumerate(samples):
        results_3 = [run_results[k][i]["result"] for k in range(NUM_RUNS)]
        expls_3   = [run_results[k][i]["explanation"] for k in range(NUM_RUNS)]
        raw_3     = [run_results[k][i]["raw_output"] for k in range(NUM_RUNS)]
        content_3 = [run_results[k][i]["content"] for k in range(NUM_RUNS)]

        counter = Counter(results_3)
        majority = counter.most_common(1)[0][0]

        final_rows.append({
            "question":               get_question(item),
            "gold":                   get_gold(item),
            "model_answer":           get_predicted(item),

            "judge_answer":           majority,
            "judge_results":          results_3,
            "judge_explanation":      expls_3[0],
            "judge_raw_output":       raw_3[0],

            "judge_explanations_3":   expls_3,
            "judge_raw_outputs_3":    raw_3,
            "judge_contents_3":       content_3,
        })

    out_path = os.path.join(output_dir, f"{base_name}_{judge_name}_judged.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total       = len(final_rows)
    consistent  = sum(1 for r in final_rows if len(set(r["judge_results"])) == 1)
    all_correct = sum(1 for r in final_rows
                      if r["judge_results"] == ["correct"] * NUM_RUNS)
    all_incorr  = sum(1 for r in final_rows
                      if r["judge_results"] == ["incorrect"] * NUM_RUNS)
    errors      = sum(1 for r in final_rows
                      if any("ERROR" in str(x) for x in r["judge_results"]))

    print(f"  ✅ 保存: {out_path}")
    print(f"     三轮一致    : {consistent}/{total}")
    print(f"     全 correct  : {all_correct}")
    print(f"     全 incorrect: {all_incorr}")
    print(f"     含 ERROR    : {errors}")

# =====================================================
# ★ 主程序
# =====================================================

def main():
    print("=" * 80)
    print(f"数据集: HQ (vanilla prompt)")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = [f for f in TARGET_FILES
                   if os.path.exists(os.path.join(INPUT_DIR, f))]
    print(f"找到 {len(input_files)} 个输入文件\n")

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"🚀 Judge: {judge_name}  workers={cfg['max_workers']}")

        for rel_path in input_files:
            input_path = os.path.join(INPUT_DIR, rel_path)
            process_one_file(input_path, OUTPUT_DIR, judge_name, cfg, client)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()