#!/usr/bin/env python3
"""
DeepSeek API judge：带推理链（with reasoning prompt） + HQ
输入: Output_WithReasoning
输出: DeepSeek_NQ_Judge_Results/WithReasoning/hq/
"""
import json
import os
import time
import re
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =====================================================
# ★ 配置
# =====================================================

BASE_URL = "https://api.deepseek.com"
API_KEY = " "
MODEL_NAME = "deepseek-v4-flash"
TEMPERATURE = 0.7
MAX_WORKERS = 48
SLEEP_INTERVAL = 0.1

BASE_DIR = r"D:\LeStoreDownload\homework\VScode\LLM_as_Judge"
INPUT_FOLDER = os.path.join(BASE_DIR, "Output_WithReasoning")
OUTPUT_DIR = os.path.join(BASE_DIR, "DeepSeek_NQ_Judge_Results", "WithReasoning", "hq")

# ★ HQ 输入文件，按需取消注释
FILENAMES = [
    "hq_DeepSeek-V4-Flash.jsonl",
    "hq_Qwen3-8B.jsonl",
    "hq_Qwen3-14B.jsonl",
    "hq_Qwen3-32B.jsonl",
    "hq_Qwen38_27b.jsonl",
]

JUDGE_MODEL = "DeepSeek-V4-Flash"
NUM_RUNS = 3

# =====================================================
# ★ Judge Prompt：带推理链
# =====================================================

JUDGE_PROMPT = """Judge whether [model_answer], the response to [question], is correct based on [gold_answer], the reference answer.
[model_reasoning] contains the reasoning produced by the model when generating [model_answer].

[question]: {question}

[model_reasoning]: {model_reasoning}

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
# ★ 辅助函数
# =====================================================

def build_judge_prompt(question, gold, model_answer, reasoning_chain):
    return JUDGE_PROMPT.format(
        question=question,
        model_answer=model_answer,
        gold_answer=gold,
        model_reasoning=reasoning_chain if reasoning_chain else "(No reasoning provided)",
    )


def extract_reasoning_chain(response_list):
    """从 response 列表中提取 <think> 内容"""
    if not response_list:
        return ""
    response_text = (
        response_list[0] if isinstance(response_list, list) else response_list
    )
    match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_judgment_and_thinking(response_text):
    for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        response_text = response_text.replace(token, "")
    response_text = response_text.strip()

    reasoning_match = re.search(
        r"Reasoning:\s*(.+?)(?=\nFinal Judgment:|$)",
        response_text,
        re.DOTALL,
    )
    judgment_match = re.search(
        r"Final Judgment:\s*(correct|incorrect)",
        response_text.lower(),
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = response_text

    if judgment_match:
        judgment = judgment_match.group(1)
    else:
        labels = re.findall(r"\b(correct|incorrect)\b", response_text.lower())
        judgment = labels[-1] if labels else "incorrect"

    thinking = f"Reasoning: {reasoning}\nFinal Judgment: {judgment}"
    return thinking, judgment


def call_deepseek(prompt, max_retries=3):
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=60.0,
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def process_one_sample(item):
    """处理单个样本（With_Reasoning 数据格式）"""
    gold = item.get("gold", "")
    if not gold:
        item["judge_result"] = "incorrect"
        item["judge_thinking"] = "ERROR: missing gold"
        return item

    model_answer = item.get("model_answer", "")
    if not model_answer:
        model_answer = "(empty)"

    response_raw = item.get("response", [])
    reasoning_chain = extract_reasoning_chain(response_raw)

    prompt = build_judge_prompt(
        question=item["question"],
        gold=gold,
        model_answer=model_answer,
        reasoning_chain=reasoning_chain,
    )

    try:
        response_text = call_deepseek(prompt)
        thinking, judgment = extract_judgment_and_thinking(response_text)
        item["judge_thinking"] = thinking
        item["judge_result"] = judgment
    except Exception as e:
        item["judge_result"] = "incorrect"
        item["judge_thinking"] = f"ERROR: {e}"

    return item

# =====================================================
# ★ 处理单个文件（一轮）
# =====================================================

def process_file_api(input_path, output_path):
    if os.path.exists(output_path):
        print(f"  ⏭️ 输出文件已存在，跳过: {output_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    total = len(samples)
    print(f"  📂 读取 {total} 条样本")

    sem = threading.Semaphore(MAX_WORKERS)

    def process_with_sem(item):
        with sem:
            time.sleep(SLEEP_INTERVAL)
            return process_one_sample(item)

    results = [None] * total
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_with_sem, samples[i]): i
            for i in range(total)
        }
        with tqdm(total=total, desc="  推理进度") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"\n    ❌ 样本 {idx+1} 处理失败: {e}")
                    err_item = samples[idx].copy()
                    err_item["judge_result"] = "incorrect"
                    err_item["judge_thinking"] = f"ERROR: {e}"
                    results[idx] = err_item
                pbar.update(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            if item is not None:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    correct_cnt = sum(
        1 for r in results if r is not None and r.get("judge_result") == "correct"
    )
    print(f"  ✅ 保存: {output_path} ({correct_cnt}/{total} correct)")

# =====================================================
# ★ 合并三轮结果
# =====================================================

def merge_runs(base_filename):
    run_files = [
        os.path.join(OUTPUT_DIR, f"{JUDGE_MODEL}_{base_filename}_run{run}.jsonl")
        for run in range(1, NUM_RUNS + 1)
    ]

    if not all(os.path.exists(f) for f in run_files):
        print(f"  ⚠️ 合并失败：缺少run文件，跳过 {JUDGE_MODEL}_{base_filename}")
        return

    merged_path = os.path.join(OUTPUT_DIR, f"{JUDGE_MODEL}_{base_filename}.jsonl")
    if os.path.exists(merged_path):
        print(f"  ⏭️ 合并文件已存在，跳过：{merged_path}")
        return

    runs_data = []
    for f in run_files:
        with open(f, 'r', encoding='utf-8') as fp:
            runs_data.append([json.loads(line) for line in fp if line.strip()])

    if not (len(runs_data[0]) == len(runs_data[1]) == len(runs_data[2])):
        print("  ⚠️ 警告：三轮样本数不一致，无法合并！")
        return

    merged = []
    for idx in range(len(runs_data[0])):
        base_item = runs_data[0][idx].copy()
        results = [
            runs_data[0][idx].get("judge_result", "incorrect"),
            runs_data[1][idx].get("judge_result", "incorrect"),
            runs_data[2][idx].get("judge_result", "incorrect"),
        ]
        thinkings = [
            runs_data[0][idx].get("judge_thinking", ""),
            runs_data[1][idx].get("judge_thinking", ""),
            runs_data[2][idx].get("judge_thinking", ""),
        ]

        base_item["judge_results"] = results
        base_item["judge_thinking"] = thinkings
        if "judge_result" in base_item:
            del base_item["judge_result"]
        merged.append(base_item)

    with open(merged_path, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  ✅ 合并完成：{merged_path}")

# =====================================================
# ★ 主程序
# =====================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"🚀 DeepSeek Judge (with reasoning, HQ, 三轮, 并发{MAX_WORKERS}线程)")
    print(f"模型: {MODEL_NAME}, 温度: {TEMPERATURE}")
    print(f"输入: {INPUT_FOLDER}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    for filename in FILENAMES:
        input_path = os.path.join(INPUT_FOLDER, filename)
        if not os.path.exists(input_path):
            print(f"⚠️ 文件不存在: {input_path}")
            continue

        base_name = filename.replace('.jsonl', '')

        for run in range(1, NUM_RUNS + 1):
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{JUDGE_MODEL}_{base_name}_run{run}.jsonl"
            )
            print(f"\n🔄 Run {run}/{NUM_RUNS}: {filename}")
            process_file_api(input_path, output_path)

        merge_runs(base_name)

    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()