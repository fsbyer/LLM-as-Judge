#!/usr/bin/env python3
"""
补跑 CF-Origin (New Gold) Judge - Qwen3.8-27B via New API
输出标准画图需要的：orign_gold_{model}_Qwen3.8-27B_judged.jsonl
"""

import json
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 配置
# =====================================================
# 读取你处理好的新 gold 目录
INPUT_DIR  = "/home/gomall/work/Counter_Factual/orign_gold_model_answer"
# 输出到画图需要的目录
OUTPUT_DIR = "/home/gomall/work/Counter_Factual/orign_gold_judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 只需要跑这 4 个 Gen 模型
TARGET_FILES = [
    "orign_context_gen_Qwen3-14B.jsonl",
    "orign_context_gen_Qwen3-32B.jsonl",
    "orign_context_gen_DeepSeek-V4.1-Flash.jsonl",
    "orign_context_gen_Qwen3.8-27B.jsonl",
]

# ★ New API 配置（沿用你之前 NQ/HQ 的配置）
NEW_API_BASE = "http://10.170.130.32:13000/v1"
NEW_API_KEY  = "sk-105QmKcyHZH1mPpkYOs3ZDtKAlMBcy4aBg6VMhFrjnja69Nu"
NEW_API_MODEL_NAME = "Qwen3.8-27B"

JUDGE_CONFIG = {
    "Qwen3.8-27B": {
        "model_name":       NEW_API_MODEL_NAME,
        "base_url":         NEW_API_BASE,
        "api_key":          NEW_API_KEY,
        "max_workers":      96,
        "enable_thinking":  True,
        "reasoning_effort": "medium",
        "temperature":      0.6,
    },
}

NUM_RUNS   = 3
MAX_TOKENS = 8192
MAX_RETRY  = 3

# =====================================================
# ★ 标准 Judge Prompt (去掉了 model_reasoning，因为反事实数据集里没有)
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
    return item.get("question", "")

def get_gold(item):
    g = item.get("gold", "")
    if isinstance(g, list):
        g = " / ".join(str(x) for x in g)
    return g

def get_predicted(item):
    # ★ 反事实数据集里，模型的回答字段叫 model_answer_orig
    return item.get("model_answer_orig") or ""

def build_prompt(item):
    return JUDGE_PROMPT.format(
        question=get_question(item),
        model_answer=get_predicted(item),
        gold_answer=get_gold(item),
    )

# =====================================================
# ★ 解析 judge 输出
# =====================================================
def parse_judge_output(raw):
    if not raw:
        return "", "", "incorrect"

    full_text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        full_text = full_text.replace(tok, "")
    full_text = full_text.strip()

    if "</think>" in full_text:
        text = full_text.split("</think>", 1)[1].strip()
    else:
        text = full_text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)", text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    m = re.search(r"Final Judgment:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        result = m.group(1).lower()
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", text.lower())
        result = labels[-1] if labels else "incorrect"

    return full_text, explanation, result

# =====================================================
# ★ 单次 API 调用
# =====================================================
def call_judge_once(client, cfg, prompt):
    kwargs = {
        "model":       cfg["model_name"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": cfg["temperature"],
        "max_tokens":  MAX_TOKENS,
    }
    if cfg["enable_thinking"]:
        kwargs["extra_body"] = {
            "reasoning_effort": cfg.get("reasoning_effort", "medium")
        }

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    msg = choice.message
    content = msg.content or ""

    reasoning = ""
    for field in ["reasoning", "reasoning_content", "thinking"]:
        v = getattr(msg, field, None)
        if v:
            reasoning = v
            break
    if not reasoning:
        try:
            d = msg.model_dump()
            for field in ["reasoning", "reasoning_content", "thinking"]:
                if d.get(field):
                    reasoning = d[field]
                    break
        except Exception:
            pass

    raw_output = (
        f"<think>\n{reasoning}\n</think>\n\n{content}"
        if reasoning else content
    )
    return raw_output, finish_reason

# =====================================================
# ★ 单轮执行
# =====================================================
def run_one_round(samples, client, cfg, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        prompt = build_prompt(item)
        last_raw = last_exp = ""
        last_res = "ERROR"
        last_err = None

        for attempt in range(MAX_RETRY + 1):
            try:
                raw_output, finish_reason = call_judge_once(client, cfg, prompt)
                judge_raw, explanation, result = parse_judge_output(raw_output)
                last_raw, last_exp, last_res = judge_raw, explanation, result
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            print(f"  ⚠️ API 失败 (idx={idx}): {last_err}", flush=True)
            last_raw = f"ERROR: {last_err}"
            last_exp = f"ERROR: {last_err}"
            last_res = "ERROR"

        new_item = item.copy()
        new_item["judge_raw_output"]  = last_raw
        new_item["judge_explanation"] = last_exp
        new_item["judge_result"]      = last_res
        return idx, new_item

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {executor.submit(process_one, i, item): i for i, item in enumerate(samples)}
        with tqdm(total=total, desc=desc) as pbar:
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"  ⚠️ future 失败: {e}", flush=True)
                pbar.update(1)

    for i, r in enumerate(results):
        if r is None:
            ph = samples[i].copy()
            ph["judge_raw_output"]  = "FATAL: result is None"
            ph["judge_explanation"] = "FATAL: result is None"
            ph["judge_result"]      = "ERROR"
            results[i] = ph

    return results

# =====================================================
# ★ 合并 3 轮
# =====================================================
def merge_runs(run_files, merged_path):
    runs = []
    for f in run_files:
        if not os.path.exists(f):
            print(f"  ⚠️ 缺少 run 文件: {f}")
            return
        with open(f, encoding="utf-8") as fp:
            runs.append([json.loads(line) for line in fp if line.strip()])

    n = len(runs[0])
    if not all(len(r) == n for r in runs):
        print(f"  ⚠️ 三轮行数不一致: {[len(r) for r in runs]}")
        return

    merged = []
    for i in range(n):
        base = runs[0][i].copy()
        for k in ["judge_raw_output", "judge_explanation", "judge_result"]:
            base.pop(k, None)

        base["judge_results"]      = [runs[k][i].get("judge_result") for k in range(3)]
        base["judge_explanations_3"] = [runs[k][i].get("judge_explanation", "") for k in range(3)]
        base["judge_raw_outputs_3"]  = [runs[k][i].get("judge_raw_output", "") for k in range(3)]
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(merged)
    consistent    = sum(1 for m in merged if len(set(m["judge_results"])) == 1)
    errors        = sum(1 for m in merged if "ERROR" in m["judge_results"])

    print(f"  ✅ 合并: {os.path.basename(merged_path)} ({total} 条)")
    print(f"     三轮一致: {consistent}/{total}  ERROR: {errors}")

# =====================================================
# ★ 主程序
# =====================================================
def main():
    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"\n🚀 Judge: {judge_name} | {NEW_API_BASE}")

        for fname in TARGET_FILES:
            input_path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(input_path):
                print(f"  ⚠️ 输入文件不存在，跳过: {fname}")
                continue

            with open(input_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            
            # 提取模型名，如 Qwen3-14B
            model_token = fname.replace("orign_context_gen_", "").replace(".jsonl", "")
            
            # ★ 强制加上 orign_gold_ 前缀，防止覆盖老文件
            out_prefix = f"orign_gold_{model_token}_{judge_name}"
            print(f"\n📁 {out_prefix} ({len(samples)} 条)")

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_run{run}.jsonl")
                run_files.append(out_path)

                if os.path.exists(out_path):
                    n_exist = sum(1 for _ in open(out_path))
                    if n_exist == len(samples):
                        print(f"  ⏭️ run{run} 已存在，跳过")
                        continue

                results = run_one_round(samples, client, cfg, desc=f"  run{run}/{NUM_RUNS}")
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in results:
                        if item is not None:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                correct = sum(1 for r in results if r and r.get("judge_result") == "correct")
                errors  = sum(1 for r in results if r and r.get("judge_result") == "ERROR")
                print(f"  ✅ run{run}: {correct}/{len(results)} correct, {errors} errors")

            merged_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_judged.jsonl")
            merge_runs(run_files, merged_path)

if __name__ == "__main__":
    main()