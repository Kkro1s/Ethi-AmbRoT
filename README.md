# Chambi Multi-Model Benchmark Evaluation
# Chambi 多模型基准评测

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI SDK](https://img.shields.io/badge/SDK-openai-green.svg)](https://github.com/openai/openai-python)

**Language / 语言:** [English](#english) · [中文](#中文)

---

<a id="english"></a>

## English

Batch-evaluate **Chambi-style** Chinese ambiguity–value alignment benchmarks by calling multiple cloud LLMs through the **OpenAI-compatible Chat Completions API**. Predictions are appended as **JSONL** (UTF-8) with **resume support** and **per-line `fsync`**, ready for offline comparison against gold labels.

### Table of contents (English)

- [Features](#features-en)
- [Repository layout](#repository-layout-en)
- [Requirements](#requirements-en)
- [Installation](#installation-en)
- [Configuration](#configuration-en)
- [Usage](#usage-en)
- [Output format](#output-format-en)
- [Resume semantics](#resume-semantics-en)
- [Customizing prompt & dataset path](#customizing-prompt--dataset-path-en)
- [Related tooling](#related-tooling-en)
- [License](#license-en)

<a id="features-en"></a>

### Features

- **Shared eval logic** in package `ethi_ambrot`: `eval_prompt.py` (test1 **and** test2 prompts), `common_eval_utils.py` (dataset I/O, phase-aware parsing, `extract_json_object` for legacy/exports, JSONL append, done-id set for resume).
- **Two-phase evaluation** via `--phase 1` (readings / 原句+解读) and `--phase 2` **Phase 2-main** (RoT + value alignment for **two readings taken from phase 1 JSONL**). Phase 2 requires `--phase1-jsonl`; only **dual-reading** phase-1 successes (distinct A/B) are evaluated. Defaults: **`output/test1/`** and **`output/test2/<provider>.jsonl`**.
- **No gold in the model prompt**: test1 uses only `input_text`; test2 uses `input_text` + `readings[].paraphrase` (never `gold_rot` / `gold_value_alignment` in the message). Other gold stays in the JSON for offline scoring.
- **Parsing**: test1 uses a Chinese layout parser (`parse_test1_response`); test2 stores a placeholder dict with `free_text` until the final output schema is fixed; `extract_json_object` remains for tooling that still consumes JSON.
- **Durable writes**: each line is followed by `flush()` + `os.fsync()` for long batch jobs.
- **Env-driven config**: keys, base URLs, and model names from the environment; optional repo `.env` (does not override existing exports).

<a id="repository-layout-en"></a>

### Repository layout

```text
.
├── README.md
├── pyproject.toml                 # optional: pip install -e .
├── requirements.txt
├── ethi_ambrot/                   # importable package
│   ├── __init__.py
│   ├── eval_prompt.py             # PROMPT_TEST1 / PROMPT_PHASE2_MAIN + builders
│   ├── phase2_main.py             # Phase 2-main filter + parse + dimensions
│   └── common_eval_utils.py
├── scripts/
│   ├── llm/                       # multi-model benchmark inference (OpenAI-compatible)
│   │   ├── run_qwen_eval.py
│   │   ├── run_doubao_eval.py
│   │   ├── run_gpt_eval.py
│   │   └── run_glm_eval.py
│   ├── dataset/                   # Chambi → compact JSON, JSONL → JSON export
│   │   ├── generate_rot_dataset.py
│   │   └── export_predictions_json.py
│   └── evaluation/                # offline metrics vs gold
│       └── evaluate_phase2_main.py
├── data/                          # benchmark JSON (and full Chambi exports)
│   ├── Chambi_benchmark_compact.json
│   ├── Chambi.json
│   └── Chambi_rot_enhanced.json
└── output/                        # eval JSONL by phase (gitignored)
    ├── test1/                     # phase 1 — ambiguity & readings
    │   └── qwen.jsonl …
    └── test2/                     # phase 2 — RoT / value (given readings)
        └── qwen.jsonl …
```

<a id="requirements-en"></a>

### Requirements

- **Python** 3.10+ (recommended: current stable)
- Valid **API credentials** per provider and, where applicable, a **compatible base URL** from official docs (Doubao must set `DOUBAO_BASE_URL` explicitly).

<a id="installation-en"></a>

### Installation

```bash
cd /path/to/your/repo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: editable install so ``from ethi_ambrot...`` works without sys.path hacks
# pip install -e .
```

`python-dotenv` is **not** required; you may install it for convenience or rely on `export` / the simple `.env` loader in the scripts.

<a id="configuration-en"></a>

### Configuration

Never commit secrets. Use environment variables or a local `.env`.

| Provider | Variables / notes |
|----------|-------------------|
| **Qwen** | `QWEN_API_KEY` (or `QWEN_MAX_API_KEY` / `DASHSCOPE_API_KEY`); optional `QWEN_BASE_URL` (DashScope-compatible default in code); `QWEN_MODEL` |
| **Doubao** | `DOUBAO_API_KEY`, **`DOUBAO_BASE_URL` (required)**, `DOUBAO_MODEL` — URL must be taken from Volcengine / Ark docs |
| **GPT** | `OPENAI_API_KEY`, `OPENAI_MODEL`; optional `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) |
| **GLM** | `GLM_API_KEY`; `GLM_MODEL` optional (defaults to `glm-4.7-flash`, see [GLM-4.7-Flash](https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash)); optional `GLM_BASE_URL`, `GLM_THINKING`, `GLM_MAX_TOKENS` |

Optional timeouts: `QWEN_TIMEOUT`, `OPENAI_TIMEOUT`, `GLM_TIMEOUT`, `DOUBAO_TIMEOUT` (seconds).

<a id="usage-en"></a>

### Usage

Run from the **repository root** via `scripts/llm/`:

```bash
export QWEN_API_KEY="sk-..." QWEN_MODEL="qwen-max"
python scripts/llm/run_qwen_eval.py

export DOUBAO_API_KEY="..." DOUBAO_BASE_URL="https://..." DOUBAO_MODEL="..."
python scripts/llm/run_doubao_eval.py

export OPENAI_API_KEY="sk-..." OPENAI_MODEL="gpt-4o"
python scripts/llm/run_gpt_eval.py

export GLM_API_KEY="..."
# optional: GLM_MODEL=glm-4.7-flash (default) or GLM_THINKING=1 per Zhipu docs
python scripts/llm/run_glm_eval.py
```

**Shared CLI**

| Flag | Meaning |
|------|---------|
| `--dataset` | Path to benchmark JSON array (default: `data/Chambi_benchmark_compact.json` via `ethi_ambrot.common_eval_utils.DEFAULT_DATASET`) |
| `--phase` | `1` = phase1；`2` = Phase 2-main（须同时提供 `--phase1-jsonl`）。 |
| `--phase1-jsonl` | Phase1 的 JSONL；**`--phase 2` 时必填**。 |
| `--output` | JSONL path (default: `output/test<phase>/<provider>.jsonl`) |
| `--limit` | Max **new** items to process this run (skipped “done” ids do not count) |
| `--sleep` | Seconds to sleep after each API call (default `0.4`) |

```bash
python scripts/llm/run_qwen_eval.py --dataset ./data/Chambi_benchmark_compact.json --limit 50 --sleep 0.5
```

<a id="output-format-en"></a>

### Output format

One JSON object per line (UTF-8):

```json
{
  "source_chambi_id": 0,
  "input_text": "……",
  "model_name": "……",
  "raw_response": "……",
  "parsed_response": { },
  "success": true,
  "error": null,
  "eval_phase": 1
}
```

- `eval_phase`: `1` or `2`; resume only matches lines with the same phase and output file.
- `parsed_response`: phase1 fields as above; phase2 **Phase 2-main**: nested `reading_a` / `reading_b` (RoT + dimensions). Legacy rows may still have `phase2_placeholder` + `free_text`. `null` on failure.
- `success`: `true` only if the API call and phase-specific parsing succeeded; otherwise `false` and `error` carries a message.

<a id="resume-semantics-en"></a>

### Resume semantics

An id is treated as **done** (skipped on the next run) only if **both** hold:

```python
record.get("success") is True and record.get("parsed_response") is not None
```

Rows with `success: false` are **retried**. Using a different `--output` path or `--phase` (and matching default file name) starts a fresh progress set. Legacy JSONL lines without `eval_phase` are treated as phase `1` when resuming phase 1 only.

<a id="customizing-prompt--dataset-path-en"></a>

### Customizing prompt & dataset path

- **Prompts**: edit `PROMPT_TEST1_TEMPLATE` / `PROMPT_TEST2_TEMPLATE` and builders in `ethi_ambrot/eval_prompt.py` (all four runners call the same helpers via `common_eval_utils.build_user_content_for_phase`).
- **Default dataset path**: `DEFAULT_DATASET` is **`<repo>/data/Chambi_benchmark_compact.json`**; override with `--dataset` if you store files elsewhere.

<a id="related-tooling-en"></a>

### Related tooling

- **`scripts/dataset/generate_rot_dataset.py`**: Chambi → compact benchmark JSON. **`scripts/dataset/export_predictions_json.py`**: merge prediction JSONL into one JSON file. **`scripts/evaluation/evaluate_phase2_main.py`**: score Phase 2-main JSONL vs dataset gold (RoT, value, differential metrics). Examples: `python scripts/llm/run_glm_eval.py --phase 2 --phase1-jsonl output/test1/glm.jsonl`; `python scripts/evaluation/evaluate_phase2_main.py -p output/test2/glm.jsonl -d data/Chambi_benchmark_compact.json -o output/eval_reports/glm_p2.json`.

<a id="license-en"></a>

### License

Add a `LICENSE` file (e.g. MIT) before open-sourcing. For issues/PRs, include Python version, provider, model name, and **redacted** error snippets.

---

<a id="中文"></a>

## 中文

基于 **Chambi** 风格的中文歧义—价值对齐基准，通过 **OpenAI 兼容的 Chat Completions API** 批量调用多个云端 LLM，将预测以 **JSONL**（UTF-8）追加落盘，支持 **断点续跑** 与 **逐条 fsync**，便于与 gold 对齐做线下评测。

### 目录（中文）

- [功能](#功能)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [环境与密钥](#环境与密钥)
- [运行方式](#运行方式)
- [输出格式](#输出格式)
- [续跑规则](#续跑规则)
- [自定义 Prompt 与数据路径](#自定义-prompt-与数据路径)
- [相关脚本](#相关脚本)
- [许可证](#许可证)

<a id="功能"></a>

### 功能

- **统一评测逻辑**：`ethi_ambrot/eval_prompt.py` 提供测试1 / 测试2 文案；`common_eval_utils.py` 负责读数据、按 `--phase` 构造 user 内容、解析、JSONL 与续跑（含 `extract_json_object` 供导出等场景）。
- **两阶段参数**：`--phase 1` 为解读任务；`--phase 2` 为 **Phase 2-main**，须指定 `--phase1-jsonl`，仅对 phase1 中**有效双解读**样本跑 RoT/价值分析；`input_text` 以数据集中为准。默认 `output/test1/`、`output/test2/`。
- **不把 gold 写进模型输入**：phase2 仅 `input_text` + phase1 的 `reading_a` / `reading_b`；`gold_rot` 等只在 `evaluate_phase2_main.py` 中对比。
- **解析**：phase1 中文版式；phase2 为 `【解读A/B】` 下六字「社会规范…理由」半结构化解析为嵌套字段。
- **可靠落盘**：每条写入后 `flush()` + `os.fsync()`，适合长时间批量任务。
- **环境变量驱动**：密钥、Base URL、模型名从环境读取；可选用项目根 `.env`（不覆盖已有 `export`）。

<a id="仓库结构"></a>

### 仓库结构

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── ethi_ambrot/                   # 可导入包：公共评测逻辑
│   ├── __init__.py
│   ├── eval_prompt.py             # PROMPT_TEST1 / PROMPT_PHASE2_MAIN
│   ├── phase2_main.py
│   └── common_eval_utils.py
├── scripts/
│   ├── llm/
│   │   ├── run_qwen_eval.py
│   │   ├── run_doubao_eval.py
│   │   ├── run_gpt_eval.py
│   │   └── run_glm_eval.py
│   ├── dataset/
│   │   ├── generate_rot_dataset.py
│   │   └── export_predictions_json.py
│   └── evaluation/
│       └── evaluate_phase2_main.py
├── data/                          # 数据 JSON
│   ├── Chambi_benchmark_compact.json
│   ├── Chambi.json
│   └── Chambi_rot_enhanced.json
└── output/                        # 按阶段分子目录（.gitignore）
    ├── test1/
    └── test2/
```

<a id="环境要求"></a>

### 环境要求

- **Python** 3.10+（推荐当前稳定版）
- 各平台有效 **API Key**；Doubao 须按文档显式配置 **`DOUBAO_BASE_URL`**。

<a id="安装"></a>

### 安装

```bash
cd /path/to/数据集
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 可选：pip install -e .
```

`python-dotenv` **非**必需；可自行安装，或依赖 shell `export` / 脚本内简易 `.env` 解析。

<a id="环境与密钥"></a>

### 环境与密钥

**勿将密钥提交到 Git**。使用环境变量或本地 `.env`。

| 平台 | 变量 / 说明 |
|------|-------------|
| **Qwen** | `QWEN_API_KEY`（或 `QWEN_MAX_API_KEY` / `DASHSCOPE_API_KEY`）；`QWEN_BASE_URL` 有默认兼容端点；`QWEN_MODEL` |
| **Doubao** | `DOUBAO_API_KEY`、**`DOUBAO_BASE_URL`（必填）**、`DOUBAO_MODEL`（URL 以火山引擎官方为准） |
| **GPT** | `OPENAI_API_KEY`、`OPENAI_MODEL`；`OPENAI_BASE_URL` 默认 `https://api.openai.com/v1` |
| **GLM** | `GLM_API_KEY`；`GLM_MODEL` 可选（默认 `glm-4.7-flash`，见 [文档](https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash)）；另有 `GLM_BASE_URL`、`GLM_THINKING`、`GLM_MAX_TOKENS` |

可选超时：`QWEN_TIMEOUT`、`OPENAI_TIMEOUT`、`GLM_TIMEOUT`、`DOUBAO_TIMEOUT`（秒）。

<a id="运行方式"></a>

### 运行方式

在**仓库根目录**执行：评测用 `scripts/llm/`，数据脚本用 `scripts/dataset/`。

```bash
export QWEN_API_KEY="sk-..." QWEN_MODEL="qwen-max"
python scripts/llm/run_qwen_eval.py

export DOUBAO_API_KEY="..." DOUBAO_BASE_URL="https://..." DOUBAO_MODEL="..."
python scripts/llm/run_doubao_eval.py

export OPENAI_API_KEY="sk-..." OPENAI_MODEL="gpt-4o"
python scripts/llm/run_gpt_eval.py

export GLM_API_KEY="..."
# 可选: GLM_MODEL、GLM_THINKING=1（深度思考，见智谱文档）
python scripts/llm/run_glm_eval.py
```

**命令行参数（四脚本一致）**

| 参数 | 说明 |
|------|------|
| `--dataset` | 基准 JSON 路径（默认 `data/Chambi_benchmark_compact.json`） |
| `--phase` | `1`：phase1；`2`：Phase 2-main（须 `--phase1-jsonl`）。 |
| `--phase1-jsonl` | **`--phase 2` 必填**，phase1 输出 JSONL。 |
| `--output` | JSONL 路径（默认 `output/test<phase>/<厂商>.jsonl`） |
| `--limit` | 本轮**最多新处理**条数（已跳过的不计） |
| `--sleep` | 每条请求后休眠秒数（默认 `0.4`） |

```bash
python scripts/llm/run_qwen_eval.py --dataset ./data/Chambi_benchmark_compact.json --limit 50 --sleep 0.5
```

<a id="输出格式"></a>

### 输出格式

每行一条 JSON（UTF-8）：

```json
{
  "source_chambi_id": 0,
  "input_text": "……",
  "model_name": "……",
  "raw_response": "……",
  "parsed_response": { },
  "success": true,
  "error": null,
  "eval_phase": 1
}
```

- **`eval_phase`**：本条记录所属阶段；续跑时与同文件内同 `eval_phase` 的成功样本合并判断。
- **`parsed_response`**：phase1 见 `original_sentence` / 解读字段；phase2 为嵌套 `reading_a` / `reading_b`（各含 RoT 与价值维度字段）；失败为 `null`。
- **`success`**：接口成功且本阶段解析成功为 `true`；否则 `false`，**`error`** 为说明字符串。

<a id="续跑规则"></a>

### 续跑规则

仅当**同时**满足下列条件时，`source_chambi_id` 视为已完成，下次运行会跳过：

```python
record.get("success") is True and record.get("parsed_response") is not None
```

**`success: false` 的样本会重跑**；更换 `--output` 或改用不同 `--phase`（及默认文件名）即另一套进度。无 `eval_phase` 字段的旧行仅在续跑 **phase 1** 时视为阶段 1。

<a id="自定义-prompt-与数据路径"></a>

### 自定义 Prompt 与数据路径

- **Prompt**：修改 `ethi_ambrot/eval_prompt.py` 中的 `PROMPT_TEST1_TEMPLATE` / `PROMPT_TEST2_TEMPLATE`；四脚本通过 `build_user_content_for_phase` 共用。
- **默认数据路径**：`DEFAULT_DATASET` 为 **`<仓库根>/data/Chambi_benchmark_compact.json`**；其他位置请用 `--dataset`。

<a id="相关脚本"></a>

### 相关脚本

- **`scripts/dataset/generate_rot_dataset.py`**：Chambi → 精简 benchmark。
- **`scripts/dataset/export_predictions_json.py`**：评测 JSONL 合并为单个 JSON。
- **`scripts/evaluation/evaluate_phase2_main.py`**：Phase 2-main 与 gold 的离线指标（RoT、价值、A/B 区分度）。

<a id="许可证"></a>

### 许可证

公开仓库前请自行添加 `LICENSE`（如 MIT）。提 Issue / PR 时请附 Python 版本、厂商、模型名及**脱敏**报错信息，便于复现。
