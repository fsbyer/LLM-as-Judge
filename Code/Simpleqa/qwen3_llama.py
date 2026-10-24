#!/usr/bin/env python3
"""
用 OpenAI GRADER_TEMPLATE 做 judge，对 answers_split 下的文件跑 3 轮
输出 A/B/C 解析为 correct / incorrect / not_attempted
"""

import json
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 配置区
# =====================================================

INPUT_DIR   = "/home/gomall/work/SimpleQA/answers_split"
OUTPUT_BASE = "/home/gomall/work/SimpleQA/Judge"
SUB_DIR     = "openai_grader"

JUDGE_CONFIG = {
    # "Qwen3-14B": {
    #     "model_name": "/home/gomall/models/Qwen3-14B",
    #     "port": 8002,
    #     "max_workers": 128,
    #     "enable_thinking": True,
    # },
    # "Qwen3-32B": {
    #     "model_name": "/home/gomall/models/Qwen3-32B",
    #     "port": 8003,
    #     "max_workers": 64,
    #     "enable_thinking": True,
    # },
    "Llama-3.1-8B-Instruct": {
        "model_name": "/home/gomall/models/Llama-3.1-8B-Instruct",
        "port": 8004,
        "max_workers": 256,
        "enable_thinking": False,
    },
}

NUM_RUNS    = 3
TEMPERATURE = 0.6
MAX_TOKENS  = 4096

# =====================================================
# ★ OpenAI GRADER_TEMPLATE
# =====================================================

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.


These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.


These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.


These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

# =====================================================
# ★ 字段读取
# =====================================================

def get_question(item):
    return item.get("question") or item.get("problem") or ""

def get_gold(item):
    """★ 修复：兼容 answer / gold / ground_truth"""
    g = item.get("answer") or item.get("gold") or item.get("ground_truth") or ""
    if isinstance(g, list):
        g = " / ".join(str(x) for x in g)
    return g

def get_predicted(item):
    if item.get("model_answer"):
        return item["model_answer"]
    raw = item.get("raw_output", "")
    if not raw:
        return ""
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

def build_prompt(item):
    return GRADER_TEMPLATE.format(
        question=get_question(item),
        target=get_gold(item),
        predicted_answer=get_predicted(item),
    )

# =====================================================
# ★ 解析 judge 输出
# =====================================================

def parse_judge_output(raw):
    if not raw:
        return "", "incorrect", None

    text = raw
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = text.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    letter = None

    if lines and lines[-1].upper() in ("A", "B", "C"):
        letter = lines[-1].upper()
    else:
        m = re.search(r"(?:grade|judgment|answer|final)\s*[:：]\s*([ABC])\b",
                      text, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
        else:
            upper = text.upper()
            if "NOT_ATTEMPTED" in upper or "NOT ATTEMPTED" in upper:
                letter = "C"
            elif "INCORRECT" in upper:
                letter = "B"
            elif "CORRECT" in upper:
                letter = "A"
            else:
                tokens = re.findall(r"\b([ABC])\b", text)
                letter = tokens[-1].upper() if tokens else None

    mapping = {"A": "correct", "B": "incorrect", "C": "not_attempted"}
    judgment = mapping.get(letter, "incorrect")
    return text, judgment, letter

# =====================================================
# ★ 单轮
# =====================================================

def run_one_round(samples, client, model_name, max_workers,
                  enable_thinking, desc):
    total = len(samples)
    results = [None] * total

    def process_one(idx, item):
        prompt = build_prompt(item)
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }
            if enable_thinking:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""

            raw_output = (
                f"<think>\n{reasoning}\n</think>\n\n{content}"
                if reasoning else content
            )
            judge_reasoning, judge_result, judge_letter = parse_judge_output(raw_output)

        except Exception as e:
            print(f"  ⚠️ API 失败 (idx={idx}): {e}")
            raw_output     = f"ERROR: {e}"
            judge_reasoning = f"ERROR: {e}"
            judge_result    = "ERROR"
            judge_letter    = None

        new_item = item.copy()
        new_item["judge_raw_output"]    = raw_output
        new_item["judge_reasoning"]     = judge_reasoning
        new_item["judge_result"]        = judge_result
        new_item["judge_grade_letter"]  = judge_letter
        return idx, new_item

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        for k in ["judge_result", "judge_reasoning",
                  "judge_raw_output", "judge_grade_letter"]:
            base.pop(k, None)

        base["judge_results"]       = [runs[k][i].get("judge_result") for k in range(3)]
        base["judge_reasonings"]    = [runs[k][i].get("judge_reasoning", "") for k in range(3)]
        base["judge_raw_outputs"]   = [runs[k][i].get("judge_raw_output", "") for k in range(3)]
        base["judge_grade_letters"] = [runs[k][i].get("judge_grade_letter") for k in range(3)]
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total         = len(merged)
    consistent    = sum(1 for m in merged if len(set(m["judge_results"])) == 1)
    all_correct   = sum(1 for m in merged if m["judge_results"] == ["correct"] * 3)
    all_incorrect = sum(1 for m in merged if m["judge_results"] == ["incorrect"] * 3)
    all_notatt    = sum(1 for m in merged if m["judge_results"] == ["not_attempted"] * 3)
    errors        = sum(1 for m in merged if "ERROR" in m["judge_results"])

    print(f"  ✅ 合并: {merged_path} ({total} 条)")
    print(f"     三轮一致   : {consistent}/{total}")
    print(f"     全 correct : {all_correct}")
    print(f"     全 incorrect: {all_incorrect}")
    print(f"     全 not_attempted: {all_notatt}")
    print(f"     含 ERROR   : {errors}")

# =====================================================
# ★ 主程序
# =====================================================

def main():
    print("=" * 80)
    print(f"OpenAI GRADER judge ({NUM_RUNS} 轮, temp={TEMPERATURE})")
    print(f"输出子目录: {SUB_DIR}")
    print("=" * 80)

    TARGET_FILES = [
        "chinese_simpleqa_Qwen3.8-27B.jsonl",
        "simpleqa_verified_Qwen3.8-27B.jsonl",
        "chinese_simpleqa_500_deepseek-v4.1-flash.jsonl",
        "simpleqa_verified_500_deepseek-v4.1-flash.jsonl",
    ]
    input_files = [f for f in TARGET_FILES
                   if os.path.exists(os.path.join(INPUT_DIR, f))]
    print(f"找到 {len(input_files)} 个输入文件\n")

    judge_dir = os.path.join(OUTPUT_BASE, SUB_DIR)
    os.makedirs(judge_dir, exist_ok=True)

    for judge_name, config in JUDGE_CONFIG.items():
        client = OpenAI(
            base_url=f"http://localhost:{config['port']}/v1",
            api_key="EMPTY",
        )
        print(f"\n🚀 Judge: {judge_name} "
              f"(port {config['port']}, workers {config['max_workers']}, "
              f"thinking={config['enable_thinking']})")

        for filename in input_files:
            input_path = os.path.join(INPUT_DIR, filename)
            with open(input_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]

            prefixed = f"{judge_name}_{filename}"
            print(f"\n📁 {prefixed}  ({len(samples)} 条)")

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = os.path.join(
                    judge_dir, prefixed.replace(".jsonl", f"_run{run}.jsonl")
                )
                run_files.append(out_path)

                if os.path.exists(out_path):
                    print(f"  ⏭️ run{run} 已存在，跳过")
                    continue

                results = run_one_round(
                    samples, client, config["model_name"],
                    config["max_workers"], config["enable_thinking"],
                    desc=f"  run{run}/{NUM_RUNS}",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in results:
                        if item is not None:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                correct = sum(1 for r in results
                              if r and r.get("judge_result") == "correct")
                print(f"  ✅ run{run}: {correct}/{len(results)} correct")

            merged_path = os.path.join(judge_dir, prefixed)
            merge_runs(run_files, merged_path)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()


