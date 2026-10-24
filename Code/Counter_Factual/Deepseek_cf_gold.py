#!/usr/bin/env python3
"""
Counter-Factual Judge (DeepSeek via OpenRouter) - 只跑 orig_context_gen
- 4 个 gen 的 orig 文件
- 每个文件跑 3 轮
- 每轮独立 run 文件，保留 judge_raw_output（含完整 thinking）
- merge 可选，默认关（DO_MERGE=False）
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
# ★ 配置区
# =====================================================
BASE_DIR   = Path(r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Counter_Factual")
ORIG_DIR   = BASE_DIR / "orign_context_gen"
OUTPUT_DIR = BASE_DIR / "judge_deepseek_cf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DO_MERGE = False   # ← 想 merge 改成 True

INPUT_FILES = [
    {"model": "Qwen3-14B", "tag": "orig",
     "path": ORIG_DIR / "orign_context_gen_Qwen3-14B.jsonl"},

    {"model": "Qwen3-32B", "tag": "orig",
     "path": ORIG_DIR / "orign_context_gen_Qwen3-32B.jsonl"},

    {"model": "DeepSeek-V4.1-Flash", "tag": "orig",
     "path": ORIG_DIR / "orign_context_gen_DeepSeek-V4.1-Flash.jsonl"},

    {"model": "Qwen3.8-27B", "tag": "orig",
     "path": ORIG_DIR / "orign_context_gen_Qwen3.8-27B.jsonl"},
]

JUDGE_CONFIG = {
    "DeepSeek-V4.1-Flash": {
        "model_name":      "deepseek/deepseek-v4.1-flash",
        "base_url":        "https://openrouter.ai/api/v1",
        "api_key":         "  ",
        "max_workers":     32,
        "enable_thinking": True,
        "temperature":     0.0,
        "max_tokens":      8192,
        "backend":         "openrouter",
    },
}

NUM_RUNS        = 3
REQUEST_TIMEOUT = 600
MAX_RETRY       = 3

# =====================================================
# ★ CF Judge Prompt
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
# ★ 解析 judge 输出
# =====================================================
def parse_judge_output(raw_text):
    if not raw_text:
        return "", "incorrect", True

    text = raw_text
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    text = text.replace("</think>", "").strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    m = re.search(r"Final Judgment:\s*(correct|incorrect)", text, re.IGNORECASE)
    if m:
        result = m.group(1).lower()
        parse_error = False
    else:
        result = "incorrect"
        parse_error = True

    return explanation, result, parse_error

# =====================================================
# ★ 单次 API 调用（三重兜底提取 reasoning）
# =====================================================
def call_judge_once(client, cfg, prompt):
    kwargs = {
        "model":       cfg["model_name"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens":  cfg["max_tokens"],
        "timeout":     REQUEST_TIMEOUT,
    }

    if cfg.get("backend") == "openrouter" and cfg.get("enable_thinking"):
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}
    elif cfg.get("backend") == "vllm" and cfg.get("enable_thinking"):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    msg = choice.message
    content = getattr(msg, "content", "") or ""

    # ---- 三重兜底 ----
    reasoning_field = ""
    extra = getattr(msg, "model_extra", None) or {}
    if isinstance(extra, dict):
        reasoning_field = extra.get("reasoning") or extra.get("reasoning_content") or ""
    if not reasoning_field:
        reasoning_field = (getattr(msg, "reasoning_content", None)
                           or getattr(msg, "reasoning", None) or "")
    if not reasoning_field:
        try:
            d = msg.model_dump()
            reasoning_field = d.get("reasoning") or d.get("reasoning_content") or ""
        except Exception:
            pass

    raw_output = (f"<think>\n{reasoning_field}\n</think>\n\n{content}"
                  if reasoning_field else content)
    return raw_output, finish_reason

# =====================================================
# ★ 单条处理
# =====================================================
def process_one(idx, item, client, cfg, ans_key):
    prompt = build_prompt(item, ans_key)

    last_raw           = ""
    last_finish_reason = None
    last_error         = None
    last_expl          = ""
    last_result        = "incorrect"

    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw_output, finish_reason = call_judge_once(client, cfg, prompt)
            explanation, result, parse_error = parse_judge_output(raw_output)

            last_raw           = raw_output
            last_finish_reason = finish_reason
            last_expl          = explanation
            last_result        = result

            if finish_reason == "length":
                last_error = "truncated"
                continue
            if parse_error:
                last_error = "parse_error"
                continue

            new_item = item.copy()
            new_item["judge_raw_output"]    = raw_output
            new_item["judge_explanation"]   = explanation
            new_item["judge_result"]        = result
            new_item["judge_parse_error"]   = False
            new_item["judge_finish_reason"] = finish_reason
            new_item["judge_error"]         = None
            new_item["judge_attempts"]      = attempt
            return idx, new_item

        except Exception as e:
            last_error = f"API error: {e}"
            time.sleep(min(2 ** attempt, 10))
            continue

    new_item = item.copy()
    new_item["judge_raw_output"]    = last_raw if last_raw else f"ERROR: {last_error}"
    new_item["judge_explanation"]   = last_expl
    new_item["judge_result"]        = last_result
    new_item["judge_parse_error"]   = True
    new_item["judge_finish_reason"] = last_finish_reason
    new_item["judge_error"]         = last_error
    new_item["judge_attempts"]      = MAX_RETRY
    return idx, new_item

# =====================================================
# ★ 单轮
# =====================================================
def run_one_round(samples, client, cfg, ans_key, desc):
    total = len(samples)
    results = [None] * total

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {
            executor.submit(process_one, i, item, client, cfg, ans_key): i
            for i, item in enumerate(samples)
        }
        with tqdm(total=total, desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    ph = samples[idx].copy()
                    ph["judge_raw_output"]    = f"FATAL: {e}"
                    ph["judge_explanation"]   = f"FATAL: {e}"
                    ph["judge_result"]        = "incorrect"
                    ph["judge_parse_error"]   = True
                    ph["judge_finish_reason"] = None
                    ph["judge_error"]         = f"FATAL: {e}"
                    ph["judge_attempts"]      = 0
                    results[idx] = ph
                pbar.update(1)

    for i, r in enumerate(results):
        if r is None:
            ph = samples[i].copy()
            ph["judge_raw_output"]    = "FATAL: result is None"
            ph["judge_explanation"]   = "FATAL: result is None"
            ph["judge_result"]        = "incorrect"
            ph["judge_parse_error"]   = True
            ph["judge_finish_reason"] = None
            ph["judge_error"]         = "FATAL: result is None"
            ph["judge_attempts"]      = 0
            results[i] = ph

    return results

# =====================================================
# ★ 合并（可选）
# =====================================================
def merge_runs(run_files, merged_path):
    runs = []
    for f in run_files:
        if not f.exists():
            print(f"  ⚠️ 缺少 run 文件: {f.name}，跳过合并")
            return
        with open(f, encoding="utf-8") as fp:
            runs.append([json.loads(line) for line in fp if line.strip()])

    n = len(runs[0])
    if not all(len(r) == n for r in runs):
        print(f"  ⚠️ 三轮行数不一致: {[len(r) for r in runs]}，跳过合并")
        return

    merged = []
    for i in range(n):
        base = runs[0][i].copy()
        for k in ["judge_raw_output", "judge_explanation", "judge_result",
                  "judge_parse_error", "judge_finish_reason",
                  "judge_error", "judge_attempts"]:
            base.pop(k, None)

        results_3 = [runs[k][i].get("judge_result", "incorrect")
                     for k in range(3)]
        parse_err_3 = [runs[k][i].get("judge_parse_error", True)
                       for k in range(3)]
        valid = [results_3[k] for k in range(3) if not parse_err_3[k]]

        if valid:
            majority = Counter(valid).most_common(1)[0][0]
        else:
            majority = "incorrect"

        base["judge_answer"]         = majority
        base["judge_results"]        = results_3
        base["judge_parse_error_3"]  = parse_err_3
        base["judge_explanations_3"] = [runs[k][i].get("judge_explanation", "")
                                        for k in range(3)]
        base["judge_raw_outputs_3"]  = [runs[k][i].get("judge_raw_output", "")
                                        for k in range(3)]
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_err = sum(1 for m in merged if all(m["judge_parse_error_3"]))
    n_correct = sum(1 for m in merged if m["judge_answer"] == "correct")
    print(f"  ✅ 合并: {merged_path.name} ({len(merged)} 条)")
    print(f"     三轮全 parse_error: {n_err}/{len(merged)}")
    print(f"     多数 correct: {n_correct}/{len(merged)}")

# =====================================================
# ★ 主程序
# =====================================================
def main():
    print("=" * 80)
    print(f"CF Judge (DeepSeek via OpenRouter, {NUM_RUNS} 轮) - 仅 orig")
    print(f"输出: {OUTPUT_DIR}")
    print(f"DO_MERGE = {DO_MERGE}")
    print("=" * 80)

    existing = [f for f in INPUT_FILES if f["path"].exists()]
    missing  = [f for f in INPUT_FILES if not f["path"].exists()]
    print(f"\n待跑 {len(existing)} 个文件:")
    for f in existing:
        print(f"  [{f['tag']:>4}] {f['model']:<25} {f['path'].name}")
    if missing:
        print(f"\n⚠️ 缺失 {len(missing)} 个:")
        for f in missing:
            print(f"  [{f['tag']:>4}] {f['model']:<25} {f['path'].name}")

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"\n🚀 Judge: {judge_name}  workers={cfg['max_workers']}  "
              f"thinking={cfg['enable_thinking']}  temp={cfg['temperature']}")

        for f in existing:
            input_path = f["path"]
            tag        = f["tag"]
            model_tag  = f["model"]

            with open(input_path, encoding="utf-8") as fp:
                samples = [json.loads(line) for line in fp if line.strip()]

            base_name = input_path.stem
            ans_key = "model_answer_orig" if tag == "orig" else "model_answer_cf"

            print(f"\n📁 [{tag}] {base_name}  ({len(samples)} 条)")

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = OUTPUT_DIR / f"{base_name}_{judge_name}_run{run}.jsonl"
                run_files.append(out_path)

                # 断点：已存在且条数对，跳过
                if out_path.exists():
                    with open(out_path, encoding="utf-8") as fp:
                        cnt = sum(1 for line in fp if line.strip())
                    if cnt == len(samples):
                        print(f"  ⏭️ run{run} 已存在，跳过")
                        continue

                results = run_one_round(
                    samples, client, cfg, ans_key,
                    desc=f"  run{run}/{NUM_RUNS}",
                )
                with open(out_path, "w", encoding="utf-8") as fp:
                    for item in results:
                        if item is not None:
                            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

                n_correct  = sum(1 for r in results if r.get("judge_result") == "correct")
                n_parseerr = sum(1 for r in results if r.get("judge_parse_error"))
                n_error    = sum(1 for r in results
                                 if (r.get("judge_raw_output") or "").startswith(("ERROR:", "FATAL:")))
                print(f"  ✅ run{run}: correct={n_correct}/{len(results)}  "
                      f"parse_error={n_parseerr}  api_error={n_error}")

            if DO_MERGE:
                merged_path = OUTPUT_DIR / f"{base_name}_{judge_name}_merged.jsonl"
                merge_runs(run_files, merged_path)

    print("\n" + "=" * 80)
    print("✅ 全部完成")
    print(f"   输出目录: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()