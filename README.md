# LLM-as-Judge


## 📂 文件夹命名含义

本仓库按 **评测基准数据集 (Benchmark)** 划分了不同的顶级文件夹：

| 文件夹名 | 含义描述 |
| :--- | :--- |
| `HQ_Judge_Result` | **High Quality** 数据集（无推理链，No Reasoning）的评测结果。 |
| `NQ_Judge_Result` | **Natural Questions** 数据集的评测结果（无推理链）。 |
| `C-Simpleqa_Judge_Result` | **中文 SimpleQA** 数据集的评测结果（包含推理链，With Reasoning）。 |
| `Simpleqa_Judge_Result` | **英文 SimpleQA** 数据集的评测结果。 |
| `BrowseCompPlus_Judge_Result` | **BrowseCompPlus** （搜索/浏览类问答）数据集的评测结果。 |
| `Counter_Factual_Judge_Result` | **反事实 (Counter-Factual)** 测试数据集结果（包含 Gold 正确/错误、随机随机等子集）。 |
| `Hack_Judge_Result` | **Hack 攻击/消融实验** 结果（包含 `override`, `combined`, `fake`, `disregard` 等注入攻击）。 |
| `Rubric_Natural_Judge_Result` | 基于 **Rubric (评分标准)** 的自然问题评测结果。 |
| `Dataset_Human_Label` | **人工标注/真实标签** 数据集，用于计算 Judge 模型的准确率和对齐度（包含 `is_correct` 字段）。 |
| `Feishu` | 相关飞书文档或实验记录。 |

---

## 📄 文件命名规范

在大多数 `_Judge_Result` 文件夹内部，文件命名遵循统一的规范：
👉 **`[Judge模型名]_[数据集详情]_[Gen生成模型名].jsonl`**

**示例**：`GPT-6-Luna_hq_Qwen3-14B.jsonl`
*   **Judge**：`GPT-6-Luna`（执行评判的模型）
*   **Dataset**：`hq`（hotpot question 数据集）
*   **Gen**：`Qwen3-14B`（被测试的生成模型）

**特殊后缀说明（常见于 Hack 或消融实验）**：
*   `_upper` / `_lower`：大小写攻击/提示词位置攻击的变体。
*   `_override` / `_combined` / `_fake` / `_disregard`：不同的 Prompt 注入/覆盖消融实验类型。
*   `_correct` / `_incorrect`：原始样本的 Ground Truth 是对还是错。

---

## 📊 JSONL 字段说明

每一行 `.jsonl` 文件代表一道题目的完整评测记录。核心字段解析如下：

### 基础信息字段
*   **`question`** *(String)*: 原始问题。
*   **`gold`** *(String/List)*: 标准答案 (Ground Truth)。
*   **`model_answer`** *(String)*: 生成模型 (Gen) 给出的最终答案。
*   **`response`** / `model_thinking` *(String)*: 生成模型的完整推理过程 (Reasoning Chain)。
*   **`question_id`** *(String)*: 题目唯一标识符（用于关联合并数据）。

### 裁判判定字段 (核心)
*   **`judge_results`** *(List)*: 长度为 3 的列表，表示 3 轮独立投票的最终判定（如 `["correct", "correct", "incorrect"]`）。
*   **`judge_raw_outputs`** *(List)*: 长度为 3 的列表，裁判模型生成的最原始输出（包含 `<think>` 标签等）。
*   **`judge_explanations`** / `judge_reasonings` *(List)*: 长度为 3 的列表，裁判模型对判定给出的解释。
*   **`judge_is_correct`** *(Boolean)*: *(部分合并文件中有)* 综合 3 轮结果得出的最终单一对错判定。

### 攻击/消融实验专有字段 (Hack 数据集)
*   **`hacked_model_answer`** *(String)*: 注入了 Hack 指令后的模型答案。
*   **`case`** / `sample_direction` *(String)*: 记录该条样本属于哪种攻击类型（如 `override_upper` 等）。

---

## 💡 如何计算 Alignment (对齐率)
如果需要计算 Judge 模型与人工标签（`Dataset_Human_Label`）的 Alignment：
1. 使用 `question_id` 或 `question` 字段将两份数据进行 Join。
2. 比对 `judge_results`（按索引逐轮比对）与人工标签中的 `is_correct`（True/False）。
3. 统计 500 题 × 3 轮 = 1500 个点的命中率，即可得到每一轮的准确率。
