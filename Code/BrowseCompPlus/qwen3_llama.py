"""
Judge 脚本：GRADER_TEMPLATE (yes/no)
对 glm/gpt/qwen 三个 jsonl 各跑 3 轮

"""

import argparse
import json
import os
import re
import time
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =====================================================
# ★ 配置区
# =====================================================

INPUT_DIR  = "/home/gomall/work/Search_Agent"
OUTPUT_DIR = "/home/gomall/work/Search_Agent/output_judge"

TARGET_FILES = [
    "glm_final_500_with_response.jsonl",
    "gpt_final_500_with_response.jsonl",
    "qwen_final_500_with_response.jsonl",
]

# 每个模型的 max_tokens 要适配它的 --max-model-len
JUDGE_CONFIG = {
    # "Qwen3-14B": {
    #     "model_name": "/home/gomall/models/Qwen3-14B",
    #     "port": 8002,
    #     "max_workers": 128,
    #     "enable_thinking": True,
    #     "max_tokens": 16384,      # 14B 有 32768 context，够
    # },
    # "Qwen3-32B": {
    #     "model_name": "/home/gomall/models/Qwen3-32B",
    #     "port": 8003,
    #     "max_workers": 64,
    #     "enable_thinking": True,
    #     "max_tokens": 8192,       # 32B context 只有 16384，留一半给 prompt
    # },
    "Llama-3.1-8B-Instruct": {
        "model_name": "/home/gomall/models/Llama-3.1-8B-Instruct",
        "port": 8004,
        "max_workers": 256,
        "enable_thinking": False,
        "max_tokens": 8192,       # 同上
    },
}

NUM_RUNS        = 3
TEMPERATURE     = 0.6
REQUEST_TIMEOUT = 600
MAX_RETRY       = 3

# =====================================================
# ★ GRADER_TEMPLATE
# =====================================================

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()

# =====================================================
# ★ 字段读取
# =====================================================

def get_question(item):
    return item.get("question") or item.get("problem") or ""

def get_gold(item):
    return item.get("gold") or item.get("answer") or item.get("ground_truth") or ""

def get_predicted(item):
    if item.get("response"):
        return item["response"]
    if item.get("model_answer"):
        return item["model_answer"]
    if item.get("model_output"):
        return item["model_output"]
    return ""

def build_prompt(item):
    return GRADER_TEMPLATE.format(
        question=get_question(item),
        response=get_predicted(item),
        correct_answer=get_gold(item),
    )

# =====================================================
# ★ 解析 judge 输出
# =====================================================

def parse_judge_output(raw_text):
    if not raw_text:
        return "None", "", False, 100, True

    text = raw_text
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        text = text.replace(tok, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        is_correct = (match.group(1).lower() == "yes")
        parse_error = False
    else:
        is_correct = False
        parse_error = True

    m = re.search(
        r"extracted_final_answer:\s*(.+?)(?:\n\s*\[?correct_answer\]?:|\n\s*reasoning:|\n\s*correct:|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    extracted_answer = m.group(1).strip() if m else "None"
    if not extracted_answer:
        extracted_answer = "None"

    m = re.search(
        r"reasoning:\s*(.+?)(?:\n\s*correct:|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    reasoning = m.group(1).strip() if m else ""

    m = re.search(r"confidence:\s*(\d+)%?", text, re.IGNORECASE)
    confidence = int(m.group(1)) if m else 100

    return extracted_answer, reasoning, is_correct, confidence, parse_error

# =====================================================
# ★ 单次 API 调用
# =====================================================

def call_judge_once(client, model_name, prompt, enable_thinking, max_tokens):
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "timeout": REQUEST_TIMEOUT,
    }
    if enable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    msg = choice.message
    content = msg.content or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    raw_output = (
        f"<think>\n{reasoning_content}\n</think>\n\n{content}"
        if reasoning_content else content
    )
    return raw_output, finish_reason

# =====================================================
# ★ 单条处理（所有情况都重试 3 次）
# =====================================================

def process_one(idx, item, client, model_name, enable_thinking, max_tokens):
    prompt = build_prompt(item)

    last_raw           = ""
    last_finish_reason = None
    last_error         = None
    last_extracted     = "None"
    last_reasoning     = ""
    last_confidence    = 0

    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw_output, finish_reason = call_judge_once(
                client, model_name, prompt, enable_thinking, max_tokens
            )
            extracted, reasoning, is_correct, confidence, parse_error = \
                parse_judge_output(raw_output)

            last_raw           = raw_output
            last_finish_reason = finish_reason
            last_extracted     = extracted
            last_reasoning     = reasoning
            last_confidence    = confidence

            if finish_reason == "length":
                last_error = "truncated"
                continue
            if parse_error:
                last_error = "parse_error"
                continue

            new_item = item.copy()
            new_item["judge_raw_output"]       = raw_output
            new_item["judge_extracted_answer"] = extracted
            new_item["judge_reasoning"]        = reasoning
            new_item["judge_is_correct"]       = is_correct
            new_item["judge_confidence"]       = confidence
            new_item["judge_parse_error"]      = False
            new_item["judge_finish_reason"]    = finish_reason
            new_item["judge_error"]            = None
            new_item["judge_attempts"]         = attempt
            return idx, new_item

        except Exception as e:
            last_error = f"API error: {e}"
            time.sleep(min(2 ** attempt, 10))
            continue

    new_item = item.copy()
    new_item["judge_raw_output"]       = last_raw if last_raw else f"ERROR: {last_error}"
    new_item["judge_extracted_answer"] = last_extracted
    new_item["judge_reasoning"]        = last_reasoning
    new_item["judge_is_correct"]       = False
    new_item["judge_confidence"]       = last_confidence
    new_item["judge_parse_error"]      = True
    new_item["judge_finish_reason"]    = last_finish_reason
    new_item["judge_error"]            = last_error
    new_item["judge_attempts"]         = MAX_RETRY
    return idx, new_item

# =====================================================
# ★ 单轮
# =====================================================

def run_one_round(samples, client, model_name, max_workers,
                  enable_thinking, max_tokens, desc):
    total = len(samples)
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one, i, item, client, model_name,
                enable_thinking, max_tokens
            ): i
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
                    ph["judge_raw_output"]       = f"FATAL: {e}"
                    ph["judge_extracted_answer"] = "None"
                    ph["judge_reasoning"]        = f"FATAL: {e}"
                    ph["judge_is_correct"]       = False
                    ph["judge_confidence"]       = 0
                    ph["judge_parse_error"]      = True
                    ph["judge_finish_reason"]    = None
                    ph["judge_error"]            = f"FATAL: {e}"
                    ph["judge_attempts"]         = 0
                    results[idx] = ph
                pbar.update(1)

    for i, r in enumerate(results):
        if r is None:
            ph = samples[i].copy()
            ph["judge_raw_output"]       = "FATAL: result is None"
            ph["judge_extracted_answer"] = "None"
            ph["judge_reasoning"]        = "FATAL: result is None"
            ph["judge_is_correct"]       = False
            ph["judge_confidence"]       = 0
            ph["judge_parse_error"]      = True
            ph["judge_finish_reason"]    = None
            ph["judge_error"]            = "FATAL: result is None"
            ph["judge_attempts"]         = 0
            results[i] = ph

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
        if not os.path.exists(f):
            print(f"  ⚠️ 缺少 run 文件: {f}，跳过合并")
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
        for k in ["judge_raw_output", "judge_extracted_answer",
                  "judge_reasoning", "judge_is_correct", "judge_confidence",
                  "judge_parse_error", "judge_finish_reason",
                  "judge_error", "judge_attempts"]:
            base.pop(k, None)

        valid_votes = []
        for k in range(3):
            if not runs[k][i].get("judge_parse_error", False):
                valid_votes.append(bool(runs[k][i].get("judge_is_correct", False)))

        if valid_votes:
            counter = Counter(valid_votes)
            majority = counter.most_common(1)[0][0]
            merged_parse_error = False
        else:
            majority = False
            merged_parse_error = True

        base["judge_is_correct"]       = majority
        base["judge_parse_error"]      = merged_parse_error
        base["judge_num_valid_votes"]  = len(valid_votes)
        base["judge_votes"]            = [
            None if runs[k][i].get("judge_parse_error", False)
            else bool(runs[k][i].get("judge_is_correct", False))
            for k in range(3)
        ]
        base["judge_extracted_answer"] = runs[0][i].get("judge_extracted_answer", "None")
        base["judge_reasoning"]        = runs[0][i].get("judge_reasoning", "")
        base["judge_confidence"]       = runs[0][i].get("judge_confidence", 100)
        base["judge_raw_output"]       = runs[0][i].get("judge_raw_output", "")
        merged.append(base)

    with open(merged_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total         = len(merged)
    all_parse_err = sum(1 for m in merged if m["judge_parse_error"])
    correct_cnt   = sum(1 for m in merged if m["judge_is_correct"])
    less_votes    = sum(1 for m in merged if m["judge_num_valid_votes"] < 3)

    print(f"  ✅ 合并: {os.path.basename(merged_path)} ({total} 条)")
    print(f"     三轮全部 parse_error : {all_parse_err}/{total}")
    print(f"     有效票 < 3 的条数    : {less_votes}/{total}")
    print(f"     多数 correct         : {correct_cnt}/{total}")

# =====================================================
# ★ 服务可用性检测
# =====================================================

def is_service_available(port, timeout=5):
    try:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
        client.models.list(timeout=timeout)
        return True
    except Exception:
        return False

# =====================================================
# ★ 主程序
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default=None,
                        help="指定这次跑哪个 judge 模型（名字和 JUDGE_CONFIG 的 key 一致）。"
                             "不指定则自动跑所有可用的")
    args = parser.parse_args()

    print("=" * 80)
    print(f"GRADER judge (yes/no, {NUM_RUNS} 轮, temp={TEMPERATURE})")
    print(f"输入: {INPUT_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 决定这次跑哪些 judge
    if args.judge:
        if args.judge not in JUDGE_CONFIG:
            print(f"❌ 未知的 judge: {args.judge}")
            print(f"   可选: {list(JUDGE_CONFIG.keys())}")
            return
        judges_to_run = {args.judge: JUDGE_CONFIG[args.judge]}
    else:
        judges_to_run = JUDGE_CONFIG

    # 服务可用性检查
    available = {}
    for name, cfg in judges_to_run.items():
        port = cfg["port"]
        print(f"\n⏳ 检查 {name} (port {port}) ...", end=" ")
        if is_service_available(port):
            print("✅ 可用")
            available[name] = cfg
        else:
            print("❌ 不可用，跳过")

    if not available:
        print("\n❌ 没有任何可用的 judge 服务，退出")
        return

    input_files = [f for f in TARGET_FILES
                   if os.path.exists(os.path.join(INPUT_DIR, f))]
    print(f"\n找到 {len(input_files)} 个输入文件")

    for judge_name, config in available.items():
        client = OpenAI(
            base_url=f"http://localhost:{config['port']}/v1",
            api_key="EMPTY",
        )
        print(f"\n{'=' * 80}")
        print(f"🚀 Judge: {judge_name} "
              f"(port {config['port']}, workers {config['max_workers']}, "
              f"thinking={config['enable_thinking']}, "
              f"max_tokens={config['max_tokens']})")
        print(f"{'=' * 80}")

        for filename in input_files:
            input_path = os.path.join(INPUT_DIR, filename)
            with open(input_path, encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]

            stem = filename.replace(".jsonl", "")
            print(f"\n📁 {filename}  ({len(samples)} 条)")

            run_files = []
            for run in range(1, NUM_RUNS + 1):
                out_path = os.path.join(
                    OUTPUT_DIR, f"{judge_name}_{stem}_run{run}.jsonl"
                )
                run_files.append(out_path)

                if os.path.exists(out_path):
                    with open(out_path, encoding="utf-8") as f:
                        existing_count = sum(1 for line in f if line.strip())
                    if existing_count == len(samples):
                        print(f"  ⏭️ run{run} 已存在且条数正确，跳过")
                        continue
                    else:
                        print(f"  ⚠️ run{run} 已存在但条数 {existing_count} != "
                              f"{len(samples)}，重跑")

                results = run_one_round(
                    samples, client, config["model_name"],
                    config["max_workers"], config["enable_thinking"],
                    config["max_tokens"],
                    desc=f"  {judge_name} run{run}/{NUM_RUNS}",
                )

                with open(out_path, "w", encoding="utf-8") as f:
                    for item in results:
                        if item is not None:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                correct   = sum(1 for r in results
                                if r and r.get("judge_is_correct") is True)
                parse_err = sum(1 for r in results
                                if r and r.get("judge_parse_error") is True)
                print(f"  ✅ run{run}: {len(results)} 条, "
                      f"correct={correct}, parse_error={parse_err}")

            merged_path = os.path.join(
                OUTPUT_DIR, f"{judge_name}_{stem}_merged.jsonl"
            )
            merge_runs(run_files, merged_path)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()