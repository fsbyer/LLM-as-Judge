#!/usr/bin/env python3
"""
Counter-Factual Judge - GPT-6-Luna 
"""

import os
import re
import json
import time
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 路径配置（Windows）
# =====================================================
BASE_DIR   = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Counter_Factual"
RANDOM_DIR = os.path.join(BASE_DIR, "absolute_random_gen")

INPUT_FILES = [
    os.path.join(RANDOM_DIR, "DeepSeek-V4.1-Flash_absolute_random_gold.jsonl"),
    os.path.join(RANDOM_DIR, "Qwen3-14B_absolute_random_gold.jsonl"),
    os.path.join(RANDOM_DIR, "Qwen3-32B_absolute_random_gold.jsonl"),
    os.path.join(RANDOM_DIR, "Qwen3.8-27B_absolute_random_gold.jsonl"),
]

# ★ GPT 专属输出文件夹（与 DS 的 absolute_random_judge 分开，方便对比）
OUTPUT_DIR = os.path.join(BASE_DIR, "absolute_random_judge_GPT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# ★ Judge 配置（GPT-6-Luna via ChatAnywhere，中等思考）
# =====================================================
JUDGE_CONFIG = {
    "GPT-6-Luna": {
        "model_name":       "gpt-6-luna",
        "base_url":         "https://api.chatanywhere.tech/v1",
        "api_key":          " ",
        "max_workers":      32,
        "temperature":      None,
        "max_tokens":       16384,
        "reasoning_effort": "medium",
    },
}

NUM_RUNS    = 3
MAX_RETRIES = 3
RETRY_DELAY = 5

# =====================================================
# ★ Judge Prompt（反事实任务版，逐字沿用）
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

def get_predicted(item, ans_key):
    return item.get(ans_key, "") or ""

def build_prompt(item, ans_key):
    return JUDGE_PROMPT.format(
        question=get_question(item),
        model_answer=get_predicted(item, ans_key),
        gold_answer=get_gold(item),
    )

# =====================================================
# ★ 解析 judge 输出（GPT 版：取最后一次 Final Judgment）
# =====================================================
def parse_judge_output(raw):
    if not raw:
        return "", "", "incorrect"

    # 原始完整文本（不删任何东西，只去特殊 token）
    full_text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        full_text = full_text.replace(tok, "")
    full_text = full_text.strip()

    # 做解析用的纯净 text
    text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    # 兼容 markdown 加粗
    text = re.sub(r"\*\*\s*(Explanation|Final Judgment)\s*:\s*\*\*",
                  r"\1:", text, flags=re.IGNORECASE)

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    # ★ 取最后一次 Final Judgment（GPT 版关键改进）
    matches = re.findall(r"Final\s*Judgment\s*:\s*\**\s*(correct|incorrect)",
                         text, re.IGNORECASE)
    if matches:
        result = matches[-1].lower()
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", text.lower())
        result = labels[-1] if labels else "incorrect"

    return full_text, explanation, result

# =====================================================
# ★ 构造 API 参数（GPT 特色）
# =====================================================
def build_kwargs(cfg, prompt):
    kwargs = {
        "model":    cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "timeout":  600,
        "max_completion_tokens": cfg["max_tokens"],
    }
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("reasoning_effort"):
        kwargs["extra_body"] = {"reasoning_effort": cfg["reasoning_effort"]}
    return kwargs

# =====================================================
# ★ 单条调用（GPT 特色：max_completion_tokens 回退 + reasoning 多字段兜底）
# =====================================================
def call_one(client, item, cfg, ans_key):
    prompt = build_prompt(item, ans_key)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = build_kwargs(cfg, prompt)

            # max_completion_tokens → max_tokens 自动回退
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                if "max_completion_tokens" in err and "not supported" in err:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = cfg["max_tokens"]
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            msg = response.choices[0].message
            content = msg.content or ""

            # reasoning 多字段兜底（GPT 各渠道位置不一致）
            reasoning = ""
            for field in ["reasoning_content", "reasoning", "thinking"]:
                v = getattr(msg, field, None)
                if v:
                    reasoning = v
                    break
            if not reasoning:
                try:
                    d = msg.model_dump()
                    for field in ["reasoning_content", "reasoning", "thinking"]:
                        if d.get(field):
                            reasoning = d[field]
                            break
                except Exception:
                    pass

            raw_output = (
                f"<think>\n{reasoning}\n</think>\n\n{content}"
                if reasoning else content
            )

            _, explanation, result = parse_judge_output(raw_output)
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
                # 调用彻底失败 → 该轮记 ERROR
                return {
                    "raw_output":  f"ERROR: {err_msg}",
                    "content":     "",
                    "explanation": f"ERROR: {err_msg}",
                    "result":      "ERROR",
                }

# =====================================================
# ★ 单轮执行
# =====================================================
def run_one_round(samples, client, cfg, ans_key, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        return idx, call_one(client, item, cfg, ans_key)

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
# ★ run 文件读写（断点续跑）
# =====================================================
def save_run_file(path, samples, results, ans_key):
    with open(path, "w", encoding="utf-8") as f:
        for item, r in zip(samples, results):
            row = {
                "question":          get_question(item),
                "gold":              get_gold(item),
                "model_answer":      get_predicted(item, ans_key),
                "judge_raw_output":  r["raw_output"],
                "judge_content":     r["content"],
                "judge_explanation": r["explanation"],
                "judge_result":      r["result"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_run_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "raw_output":  row.get("judge_raw_output", ""),
                "content":     row.get("judge_content", ""),
                "explanation": row.get("judge_explanation", ""),
                "result":      row.get("judge_result", "ERROR"),
            })
    return rows

# =====================================================
# ★ 处理一个 gen 文件 × 当前启用的 judge
# =====================================================
def run_one_file(samples, input_file, judge_name, cfg, ans_key):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    print(f"\n🚀 [{os.path.basename(input_file)}] Judge: {judge_name} | "
          f"workers={cfg['max_workers']} | max_tokens={cfg.get('max_tokens')} | "
          f"reasoning_effort={cfg.get('reasoning_effort')}")

    run_results = []
    for run in range(1, NUM_RUNS + 1):
        run_path = os.path.join(OUTPUT_DIR, f"{base_name}_{judge_name}_run{run}.jsonl")

        if os.path.exists(run_path):
            loaded = load_run_file(run_path)
            if len(loaded) == len(samples):
                print(f"  ⏭️  run{run} 已存在，跳过（{len(loaded)} 条）")
                run_results.append(loaded)
                continue
            else:
                print(f"  ⚠️  run{run} 行数不一致 ({len(loaded)} vs {len(samples)})，重跑")

        res = run_one_round(samples, client, cfg, ans_key, desc=f"  run{run}/{NUM_RUNS}")
        save_run_file(run_path, samples, res, ans_key)
        run_results.append(res)

        correct = sum(1 for r in res if r and r.get("result") == "correct")
        err     = sum(1 for r in res if r and r.get("result") == "ERROR")
        print(f"  ✅ run{run}: {correct}/{len(res)} correct | ERROR={err}")

    # 合并（原版投票：Counter.most_common）
    final_rows = []
    for i, item in enumerate(samples):
        results_3 = [run_results[k][i]["result"]      for k in range(NUM_RUNS)]
        expls_3   = [run_results[k][i]["explanation"] for k in range(NUM_RUNS)]
        raws_3    = [run_results[k][i]["raw_output"]  for k in range(NUM_RUNS)]
        majority  = Counter(results_3).most_common(1)[0][0]

        final_rows.append({
            "question":             get_question(item),
            "gold":                 get_gold(item),
            "model_answer":         get_predicted(item, ans_key),
            "judge_answer":         majority,
            "judge_results":        results_3,
            "judge_explanations_3": expls_3,
            "judge_raw_outputs":    raws_3,
        })

    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{judge_name}_judged.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total     = len(final_rows)
    errors    = sum(1 for r in final_rows if any("ERROR" in str(x) for x in r["judge_results"]))
    n_correct = sum(1 for r in final_rows if r["judge_answer"] == "correct")
    n_incorr  = sum(1 for r in final_rows if r["judge_answer"] == "incorrect")

    print(f"\n  ✅ 合并: {os.path.basename(out_path)}")
    print(f"     [荒谬测试] correct: {n_correct}/{total}")
    print(f"                incorrect: {n_incorr}/{total}")
    print(f"                含 ERROR : {errors}")

# =====================================================
# ★ 主程序
# =====================================================
def main():
    print("=" * 80)
    print("Counter-Factual Judge - GPT-6-Luna (ChatAnywhere) - medium reasoning")
    print(f"输入目录: {RANDOM_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"启用 judge: {list(JUDGE_CONFIG.keys())}")
    print("=" * 80)

    ans_key = "model_answer_orig"

    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            print(f"❌ 找不到输入文件: {input_file}")
            continue

        with open(input_file, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        if not samples:
            print(f"⚠️ 空文件，跳过: {input_file}")
            continue

        for judge_name, cfg in JUDGE_CONFIG.items():
            run_one_file(samples, input_file, judge_name, cfg, ans_key)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()