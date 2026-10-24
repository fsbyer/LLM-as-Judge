#!/usr/bin/env python3
"""
用 ChatAnywhere 的 gpt-6-luna 对 Chinese-SimpleQA 做 judge
OpenAI GRADER_TEMPLATE (A/B/C)，3 轮全存（已去掉 Qwen3-8B）
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

INPUT_DIR  = os.path.join(BASE, "SimpleQA")
OUTPUT_DIR = os.path.join(BASE, "SimpleQA", "judge_gpt6luna", "chinese_simpleqa")

TARGET_FILES = [
    os.path.join("Chinese-SimpleQA", "chinese_simpleqa_Qwen3-14B.jsonl"),
    os.path.join("Chinese-SimpleQA", "chinese_simpleqa_Qwen3-32B.jsonl"),
    os.path.join("Chinese-SimpleQA", "chinese_simpleqa_Qwen3.8-27B.jsonl"),
    os.path.join("Chinese-SimpleQA", "chinese_simpleqa_500_deepseek-v4.1-flash.jsonl"),
]

JUDGE_CONFIG = {
    "GPT-6-Luna": {
        "model_name":      "gpt-6-luna",
        "base_url":        "https://api.chatanywhere.tech/v1",
        "api_key":         "  ",     # ← 换成你的
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
# ★ GRADER_TEMPLATE（OpenAI A/B/C 版本）
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
    return (item.get("gold") or item.get("answer")
            or item.get("ground_truth") or "")

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
    return GRADER_TEMPLATE.format(
        question=get_question(item),
        target=get_gold(item),
        predicted_answer=get_predicted(item),
    )

# =====================================================
# ★ 解析 judge 输出（A/B/C）
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
# ★ 构造 API 调用参数
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
# ★ 单条调用
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

            judge_explanation, judge_result, judge_letter = parse_judge_output(raw_output)
            return {
                "raw_output":       raw_output,
                "content":          content,
                "explanation":      judge_explanation,
                "result":           judge_result,
                "letter":           judge_letter,
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
                    "letter":      None,
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
        res = run_one_round(
            samples, client, cfg,
            desc=f"  run{run}/{NUM_RUNS}"
        )
        run_results.append(res)

    final_rows = []
    for i, item in enumerate(samples):
        results_3   = [run_results[k][i]["result"] for k in range(NUM_RUNS)]
        letters_3   = [run_results[k][i]["letter"] for k in range(NUM_RUNS)]
        expls_3     = [run_results[k][i]["explanation"] for k in range(NUM_RUNS)]
        raw_3       = [run_results[k][i]["raw_output"] for k in range(NUM_RUNS)]
        content_3   = [run_results[k][i]["content"] for k in range(NUM_RUNS)]

        counter = Counter(results_3)
        majority = counter.most_common(1)[0][0]

        final_rows.append({
            "query_id":               item.get("query_id"),
            "question":               get_question(item),
            "gold":                   get_gold(item),
            "model_answer":           get_predicted(item),
            "explanation":            item.get("explanation", ""),

            "judge_answer":           majority,
            "judge_results":          results_3,
            "judge_grade_letters":    letters_3,
            "judge_explanation":      expls_3[0],
            "judge_raw_output":       raw_3[0],

            "judge_answers_3":        results_3,
            "judge_explanations_3":   expls_3,
            "judge_letters_3":        letters_3,
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
    all_notatt  = sum(1 for r in final_rows
                      if r["judge_results"] == ["not_attempted"] * NUM_RUNS)
    errors      = sum(1 for r in final_rows
                      if any("ERROR" in str(x) for x in r["judge_results"]))

    print(f"  ✅ 保存: {out_path}")
    print(f"     三轮一致        : {consistent}/{total}")
    print(f"     全 correct      : {all_correct}")
    print(f"     全 incorrect    : {all_incorr}")
    print(f"     全 not_attempted: {all_notatt}")
    print(f"     含 ERROR        : {errors}")

# =====================================================
# ★ 主程序
# =====================================================

def main():
    print("=" * 80)
    print(f"数据集: Chinese-SimpleQA (OpenAI A/B/C 模板)")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = [f for f in TARGET_FILES
                   if os.path.exists(os.path.join(INPUT_DIR, f))]
    print(f"找到 {len(input_files)} 个输入文件\n")

    for judge_name, cfg in JUDGE_CONFIG.items():
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
        print(f"🚀 Judge: {judge_name}")

        for rel_path in input_files:
            input_path = os.path.join(INPUT_DIR, rel_path)
            process_one_file(
                input_path, OUTPUT_DIR, judge_name, cfg, client
            )

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()