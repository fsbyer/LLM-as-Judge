#!/usr/bin/env python3
"""
Counter-Factual Judge (Gomall) - 仅 orign_context_gen
gold 固定 = 文件里的 gold 字段（即 cf_answer）
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
# ★ 路径配置（Gomall）
# =====================================================
BASE_DIR   = "/home/gomall/work/Counter_Factual"
ORIG_DIR   = os.path.join(BASE_DIR, "orign_context_gen")
OUTPUT_DIR = os.path.join(BASE_DIR, "judge")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# ★ 输入文件列表（5 个 = 5 模型 × orig）
#    跑过的注释掉即可
# =====================================================
INPUT_FILES = [
    # ---------- Qwen3-8B ----------
    {"model": "Qwen3-8B", "tag": "orig",
     "path": os.path.join(ORIG_DIR, "orign_context_gen_Qwen3-8B.jsonl")},

    # ---------- Qwen3-14B ----------
    {"model": "Qwen3-14B", "tag": "orig",
     "path": os.path.join(ORIG_DIR, "orign_context_gen_Qwen3-14B.jsonl")},

    # ---------- Qwen3-32B ----------
    {"model": "Qwen3-32B", "tag": "orig",
     "path": os.path.join(ORIG_DIR, "orign_context_gen_Qwen3-32B.jsonl")},

    # ---------- DeepSeek-V4.1-Flash ----------
    {"model": "DeepSeek-V4.1-Flash", "tag": "orig",
     "path": os.path.join(ORIG_DIR, "orign_context_gen_DeepSeek-V4.1-Flash.jsonl")},

    # ---------- Qwen3.8-27B ----------
    {"model": "Qwen3.8-27B", "tag": "orig",
     "path": os.path.join(ORIG_DIR, "orign_context_gen_Qwen3.8-27B.jsonl")},
]

# =====================================================
# ★ Judge 模型配置（一次只开一个，其余注释掉）
# =====================================================
JUDGE_CONFIG = {
    # "Qwen3-8B": {
    #     "model_name":      "/home/gomall/models/Qwen3-8B",
    #     "base_url":        "http://localhost:8001/v1",
    #     "api_key":         "EMPTY",
    #     "max_workers":     256,
    #     "enable_thinking": True,
    #     "temperature":     0.6,
    #     "max_tokens":      8192,
    # },
    # "Qwen3-14B": {
    #     "model_name":      "/home/gomall/models/Qwen3-14B",
    #     "base_url":        "http://localhost:8002/v1",
    #     "api_key":         "EMPTY",
    #     "max_workers":     128,
    #     "enable_thinking": True,
    #     "temperature":     0.6,
    #     "max_tokens":      8192,
    # },
    # "Qwen3-32B": {
    #     "model_name":      "/home/gomall/models/Qwen3-32B",
    #     "base_url":        "http://localhost:8003/v1",
    #     "api_key":         "EMPTY",
    #     "max_workers":     64,
    #     "enable_thinking": True,
    #     "temperature":     0.6,
    #     "max_tokens":      8192,
    # },
    "Llama-3.1-8B-Instruct": {
        "model_name":      "/home/gomall/models/Llama-3.1-8B-Instruct",
        "base_url":        "http://localhost:8004/v1",
        "api_key":         "EMPTY",
        "max_workers":     256,
        "enable_thinking": False,     # Llama 没有 think
        "temperature":     0.7,
        "max_tokens":      8192,
    },
}

NUM_RUNS    = 3
MAX_RETRIES = 3
RETRY_DELAY = 5

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
def get_question(item):
    return item.get("question", "")


def get_gold(item):
    g = item.get("gold", "")
    if isinstance(g, list):
        g = " / ".join(str(x) for x in g)
    return g


def get_predicted(item, ans_key):
    return item.get(ans_key, "") or ""


def get_reasoning(item, raw_key):
    raw = item.get(raw_key, "") or ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def build_prompt(item, ans_key):
    return JUDGE_PROMPT.format(
        question=get_question(item),
        model_answer=get_predicted(item, ans_key),
        gold_answer=get_gold(item),
    )

# =====================================================
# ★ 解析 judge 输出
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
# ★ 构造 API 参数（vLLM 走 chat_template_kwargs）
# =====================================================
def build_kwargs(cfg, prompt):
    kwargs = {
        "model":    cfg["model_name"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg.get("max_tokens") is not None:
        kwargs["max_tokens"] = cfg["max_tokens"]
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("enable_thinking"):
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True}
        }
    return kwargs

# =====================================================
# ★ 单条调用
# =====================================================
def call_one(client, item, cfg, ans_key):
    prompt = build_prompt(item, ans_key)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = build_kwargs(cfg, prompt)
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            content   = getattr(msg, "content", "") or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""

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
                return {
                    "raw_output":  f"ERROR: {err_msg}",
                    "content":     "",
                    "explanation": f"ERROR: {err_msg}",
                    "result":      "ERROR",
                }

# =====================================================
# ★ 单轮
# =====================================================
def run_one_round(samples, client, cfg, ans_key, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        r = call_one(client, item, cfg, ans_key)
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
def process_one_file(input_path, output_dir, tag, judge_name, cfg, client):
    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    ans_key = "model_answer_orig"
    raw_key = "model_raw_output_orig"

    print(f"\n📁 [{tag}] {base_name}  ({len(samples)} 条)")

    run_results = []
    for run in range(1, NUM_RUNS + 1):
        res = run_one_round(samples, client, cfg, ans_key,
                            desc=f"  run{run}/{NUM_RUNS}")
        run_results.append(res)

        run_path = os.path.join(
            output_dir, f"{base_name}_{judge_name}_run{run}.jsonl"
        )
        with open(run_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(samples):
                row = {
                    "question":          get_question(item),
                    "gold":              get_gold(item),
                    "model_answer":      get_predicted(item, ans_key),
                    "model_reasoning":   get_reasoning(item, raw_key),
                    "judge_raw_output":  res[i]["raw_output"],
                    "judge_content":     res[i]["content"],
                    "judge_explanation": res[i]["explanation"],
                    "judge_result":      res[i]["result"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        correct = sum(1 for r in res if r and r.get("result") == "correct")
        print(f"  ✅ run{run}: {correct}/{len(res)} correct")

    # 合并（多数投票）
    final_rows = []
    for i, item in enumerate(samples):
        results_3 = [run_results[k][i]["result"] for k in range(NUM_RUNS)]
        expls_3   = [run_results[k][i]["explanation"] for k in range(NUM_RUNS)]
        counter   = Counter(results_3)
        majority  = counter.most_common(1)[0][0]

        final_rows.append({
            "question":             get_question(item),
            "gold":                 get_gold(item),
            "model_answer":         get_predicted(item, ans_key),
            "model_reasoning":      get_reasoning(item, raw_key),
            "judge_answer":         majority,
            "judge_results":        results_3,
            "judge_explanations_3": expls_3,
        })

    out_path = os.path.join(output_dir,
                            f"{base_name}_{judge_name}_judged.jsonl")
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
    n_correct   = sum(1 for r in final_rows if r["judge_answer"] == "correct")

    print(f"  ✅ 合并: {os.path.basename(out_path)}")
    print(f"     [{tag}] correct(多数): {n_correct}/{total}")
    print(f"     三轮一致    : {consistent}/{total}")
    print(f"     含 ERROR    : {errors}")

    return {"tag": tag, "n_correct": n_correct, "total": total}

# =====================================================
# ★ 主程序
# =====================================================
def main():
    print("=" * 80)
    print(f"Counter-Factual Judge (Gomall, {NUM_RUNS} 轮, 仅 orig)")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    existing = [f for f in INPUT_FILES if os.path.exists(f["path"])]
    missing  = [f for f in INPUT_FILES if not os.path.exists(f["path"])]

    print(f"\n待跑 {len(existing)} 个文件:")
    for f in existing:
        print(f"  [{f['tag']:>4}] {f['model']:<25} {os.path.basename(f['path'])}")
    if missing:
        print(f"\n⚠️ 缺失 {len(missing)} 个文件:")
        for f in missing:
            print(f"  [{f['tag']:>4}] {f['model']:<25} {os.path.basename(f['path'])}")

    all_results = []

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"\n🚀 Judge: {judge_name}  workers={cfg['max_workers']}  "
              f"thinking={cfg['enable_thinking']}  temp={cfg['temperature']}")

        for f in existing:
            r = process_one_file(
                input_path=f["path"],
                output_dir=OUTPUT_DIR,
                tag=f["tag"],
                judge_name=judge_name,
                cfg=cfg,
                client=client,
            )
            r["model"] = f["model"]
            all_results.append(r)

    # ---------- 汇总 ----------
    print("\n" + "=" * 80)
    print("📊 汇总")
    print("=" * 80)
    print(f"{'Model':<25} {'orig':>12} {'acc':>8}")
    print("-" * 80)

    for r in all_results:
        acc = r["n_correct"] / r["total"] * 100 if r["total"] else 0
        print(f"{r['model']:<25} {r['n_correct']:>5}/{r['total']:<6} {acc:>7.1f}%")

    print("=" * 80)
    print("✅ 全部完成")


if __name__ == "__main__":
    main()