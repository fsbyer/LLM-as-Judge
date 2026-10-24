#!/usr/bin/env python3
"""
用 Reference-Based Rubric prompt 对 Eval_500.jsonl 做 judge
"""

import json
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 配置区 (已修改为 Windows 路径)
# =====================================================

# 输入文件路径
INPUT_DIR  = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Rubric"
TARGET_FILES = ["Eval_500.jsonl"]

# ★ 为 DS 单独建一个输出文件夹，方便和后面的 GPT 隔离
OUTPUT_DIR = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Rubric\Deepseek-V4.1-flash_Judge"

# DS 专属 API 配置
JUDGE_CONFIG = {
    "DeepSeek-V4.1-Flash": {
        "model_name":      "deepseek/deepseek-v4.1-flash",
        "base_url":        "https://openrouter.ai/api/v1",
        "api_key":         " ",
        "max_workers":     32,
        "enable_thinking": True,
        "max_tokens":      8192,
    },
}

NUM_RUNS   = 3
MAX_TOKENS = 8192

# =====================================================
# ★ 新版 Reference-Based Rubric Prompt
# =====================================================

JUDGE_PROMPT = """Judge whether [model_answer], the response to [question], is correct based on [gold_answer], the reference answer.

[question]: {question}

[model_answer]: {model_answer}

[gold_answer]: {gold_answer}

Evaluate semantic equivalence between [model_answer] and [gold_answer] in the context of [question], rather than exact wording.
If [gold_answer] provides alternative answers, matching any one complete alternative is sufficient.

Decision Rules:
1. **Reference-Based Evaluation**: Use [gold_answer] as the reference for comparison. Do not independently verify or challenge the factual correctness, timeliness, or completeness of [gold_answer]. The task is to evaluate [model_answer] against the reference answer, not to independently fact-check the reference itself.
2. **Entity Equivalence**: Different names, aliases, abbreviations, transliterations, and translated names should be treated as equivalent when they unambiguously refer to the same entity in the context of the question. Historical and modern names are equivalent only when they refer to the same relevant entity under the temporal context of the question.
3. **Completeness and Specificity**: [model_answer] must provide sufficient information to answer [question] correctly.
   - **Greater specificity**: A more specific answer is acceptable if it includes the requested information and does not conflict with [question] or contradict information explicitly stated in [gold_answer].
   - **Lower specificity**: A less specific answer is acceptable only if it still sufficiently answers the question. Do not require a level of precision beyond what the question asks for. For example, a month or year is sufficient when the question asks for the month or year, respectively; an exact-date question requires the exact date.
   - **Nonessential differences**: Do not mark an answer incorrect solely because its wording, level of detail, or degree of specificity differs from [gold_answer], provided it conveys the same answer and satisfies the question.
4. **Alternative Answers and Required Components**: If [gold_answer] explicitly provides alternative answers, matching any one complete alternative is sufficient. If [gold_answer] contains multiple jointly required components, [model_answer] must include all essential components to be considered correct.
5. **Additional Information**: Correct and relevant additional information does not invalidate an otherwise correct answer. However, materially false, contradictory, or misleading additional claims make the entire answer incorrect, even if the core answer is correct. Do not assume that additional information is false merely because it is absent from [gold_answer]. Treat it as incorrect only if it contradicts [question] or information explicitly stated in [gold_answer], or is clearly false based on well-established facts. Additional information must also satisfy the category, scope, and constraints specified in [question].

Judge [model_answer] as correct if it fulfills the above decision rules and is semantically equivalent to [gold_answer]; otherwise, judge it as incorrect.

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
    return item.get("model_answer") or ""

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

    text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = text.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    m = re.search(r"Final Judgment:\s*(correct|incorrect)",
                  text, re.IGNORECASE)
    if m:
        result = m.group(1).lower()
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", text.lower())
        result = labels[-1] if labels else "incorrect"

    return text, explanation, result

# =====================================================
# ★ 单轮
# =====================================================

def run_one_round(samples, client, cfg, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        prompt = build_prompt(item)
        try:
            # ★ DS 专属 OpenRouter 传参
            extra_body = {"reasoning": {"effort": "high"}} if cfg["enable_thinking"] else None
            
            response = client.chat.completions.create(
                model=cfg["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                extra_body=extra_body,
            )
            
            msg = response.choices[0].message
            content = msg.content or ""
            
            # ★ DS 专属 reasoning 提取逻辑
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details:
                reasoning = "\n".join(d.get("text", "") for d in reasoning_details if isinstance(d, dict))
            else:
                reasoning = getattr(msg, "reasoning_content", None) or ""

            raw_output = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
            judge_raw, explanation, result = parse_judge_output(raw_output)

        except Exception as e:
            print(f"  ⚠️ API 失败 (idx={idx}): {e}")
            judge_raw   = f"ERROR: {e}"
            explanation = f"ERROR: {e}"
            result      = "ERROR"

        new_item = item.copy()
        new_item["judge_raw_output"]  = judge_raw
        new_item["judge_explanation"] = explanation
        new_item["judge_result"]      = result
        return idx, new_item

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {executor.submit(process_one, i, item): i
                   for i, item in enumerate(samples)}
        with tqdm(total=total, desc=desc) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)

    return results

# =====================================================
# ★ 合并 3 轮
# =====================================================

def merge_runs(run_files, merged_path):
    if os.path.exists(merged_path):
        print(f"  ⏭️ 合并文件已存在: {merged_path}")
        return

    runs = []
    for f in run_files:
        with open(f, encoding="utf-8") as fp:
            runs.append([json.loads(line) for line in fp if line.strip()])

    n = len(runs[0])
    if not all(len(r) == n for r in runs):
        print(f"  ⚠️ 三轮行数不一致，跳过合并")
        return

    merged = []
    for i in range(n):
        base = runs[0][i].copy()
        for k in ["judge_raw_output", "judge_explanation", "judge_result"]:
            base.pop(k, None)

        base["judge_results"]      = [runs[k][i].get("judge_result") for k in range(3)]
        base["judge_explanations"] = [runs[k][i].get("judge_explanation", "") for k in range(3)]
        base["judge_raw_outputs"]  = [runs[k][i].get("judge_raw_output", "") for k in range(3)]
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(merged)
    consistent    = sum(1 for m in merged if len(set(m["judge_results"])) == 1)
    all_correct   = sum(1 for m in merged if m["judge_results"] == ["correct"] * 3)
    all_incorrect = sum(1 for m in merged if m["judge_results"] == ["incorrect"] * 3)
    errors        = sum(1 for m in merged if "ERROR" in m["judge_results"])

    print(f"  ✅ 合并: {merged_path} ({total} 条)")
    print(f"     三轮一致: {consistent}/{total}  全 correct: {all_correct}  "
          f"全 incorrect: {all_incorrect}  ERROR: {errors}")

# =====================================================
# ★ 主程序
# =====================================================

def main():
    print("=" * 80)
    print(f"Reference-Based Rubric judge ({NUM_RUNS} 轮)")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
            
        print(f"\n🚀 Judge: {judge_name}  workers={cfg['max_workers']}  "
              f"thinking={cfg['enable_thinking']}")

        for fname in TARGET_FILES:
            input_path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(input_path):
                print(f"  ⚠️ 不存在，跳过: {input_path}")
                continue

            with open(input_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]

            prefixed = f"{judge_name}_rubric"
            print(f"\n📁 {prefixed}  ({len(samples)} 条)")

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = os.path.join(OUTPUT_DIR, f"{prefixed}_run{run}.jsonl")
                run_files.append(out_path)

                if os.path.exists(out_path):
                    print(f"  ⏭️ run{run} 已存在，跳过")
                    continue

                results = run_one_round(samples, client, cfg,
                                        desc=f"  run{run}/{NUM_RUNS}")
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in results:
                        if item is not None:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                correct = sum(1 for r in results
                              if r and r.get("judge_result") == "correct")
                print(f"  ✅ run{run}: {correct}/{len(results)} correct")

            merged_path = os.path.join(OUTPUT_DIR, f"{prefixed}.jsonl")
            merge_runs(run_files, merged_path)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()