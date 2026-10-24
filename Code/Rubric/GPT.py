#!/usr/bin/env python3
"""
Reference-Based Rubric prompt 对 Eval_500.jsonl 做 judge，跑 3 轮
"""

import json
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 配置区 (Windows 路径)
# =====================================================

INPUT_DIR  = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Rubric"
TARGET_FILES = ["Eval_500.jsonl"]

# ★ GPT 专属输出文件夹
OUTPUT_DIR = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge\Rubric\GPT-6-Luna_judge"

# GPT 专属 API 配置
JUDGE_CONFIG = {
    "GPT-6-Luna": {
        "model_name":       "gpt-6-luna",
        "base_url":         "https://api.chatanywhere.tech/v1",
        "api_key":          "  ",
        "max_workers":      32,
        "temperature":      None,     # GPT 思考模式下通常不支持设置 temperature，设为 None 避免报错
        "max_tokens":       16384,    # 开启思考后调大 token 上限防止截断
        "reasoning_effort": "medium", # ★ 明确指定中等思考
    },
}

NUM_RUNS   = 3
MAX_TOKENS = JUDGE_CONFIG["GPT-6-Luna"]["max_tokens"]

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
# ★ 解析 judge 输出 (提取最后一次判定 + 保留原始完整文本)
# =====================================================

def parse_judge_output(raw):
    if not raw:
        return "", "", "incorrect"

    # 最原始的输出，什么都不删
    full_text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        full_text = full_text.replace(tok, "")
    full_text = full_text.strip()

    # 用来做解析的纯净版 text
    text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    m = re.search(r"Explanation:\s*(.+?)(?=\n\s*Final Judgment:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    explanation = m.group(1).strip() if m else text

    # ★ 提取所有匹配项，取最后一次
    matches = re.findall(r"Final\s*Judgment\s*:\s*\**\s*(correct|incorrect)", text, re.IGNORECASE)
    if matches:
        result = matches[-1].lower()
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", text.lower())
        result = labels[-1] if labels else "incorrect"

    # ★ 返回最原始的 full_text
    return full_text, explanation, result

# =====================================================
# ★ 单轮 (GPT 专属调用逻辑)
# =====================================================

def run_one_round(samples, client, cfg, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        prompt = build_prompt(item)
        
        kwargs = {
            "model": cfg["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "timeout": 600,
        }
        
        # 优先使用 max_completion_tokens，失败再回退
        kwargs["max_completion_tokens"] = cfg["max_tokens"]

        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
            
        # ★ 通过 extra_body 传递 reasoning_effort，兼容第三方 API
        if cfg.get("reasoning_effort"):
            kwargs["extra_body"] = {"reasoning_effort": cfg["reasoning_effort"]}

        content = ""
        reasoning = ""
        raw_output = ""
        judge_result = "ERROR"
        judge_raw = ""
        explanation = ""

        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                # 自动回退：如果不支持 max_completion_tokens
                if "max_completion_tokens" in err and "not supported" in err:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = cfg["max_tokens"]
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            msg = response.choices[0].message
            content = msg.content or ""

            # 兼容抓取 reasoning 内容（虽然可能不返回，但留着兜底）
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

            raw_output = (f"<think>\n{reasoning}\n</think>\n\n{content}"
                          if reasoning else content)
            judge_raw, explanation, judge_result = parse_judge_output(raw_output)

        except Exception as e:
            print(f"  ⚠️ API 失败 (idx={idx}): {e}")
            judge_raw   = f"ERROR: {e}"
            explanation = f"ERROR: {e}"
            judge_result = "ERROR"

        new_item = item.copy()
        new_item["judge_raw_output"]  = judge_raw
        new_item["judge_explanation"] = explanation
        new_item["judge_result"]      = judge_result
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
# ★ 合并 3 轮 (与 Qwen3 完全一致)
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
    print(f"GPT Judge ({NUM_RUNS} 轮) - {list(JUDGE_CONFIG.keys())[0]} (中等思考)")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

        print(f"\n🚀 Judge: {judge_name}  workers={cfg['max_workers']}  "
              f"reasoning_effort={cfg['reasoning_effort']}")

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