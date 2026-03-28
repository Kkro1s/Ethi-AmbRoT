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

- **Shared eval logic** in `common_eval_utils.py`: dataset I/O, prompt, `extract_json_object`, JSONL append, done-id set for resume.
- **One runner per provider**; default outputs go to separate files under `output/` (e.g. `output/qwen_predictions.jsonl`).
- **No gold in the prompt**: only `input_text` + the shared template; `ambiguity_type` / `value_dimension` / `readings` stay in the dataset for offline evaluation.
- **Robust JSON parsing**: strips \`\`\`json fences, tolerates leading/trailing chatter; falls back to the first `{` … last `}` slice before failing.
- **Durable writes**: each line is followed by `flush()` + `os.fsync()` for long batch jobs.
- **Env-driven config**: keys, base URLs, and model names from the environment; optional repo `.env` (does not override existing exports).

<a id="repository-layout-en"></a>

### Repository layout

```text
.
├── README.md
├── requirements.txt
├── common_eval_utils.py
├── run_qwen_eval.py
├── run_doubao_eval.py
├── run_gpt_eval.py
├── run_glm_eval.py
├── generate_rot_dataset.py       # optional: Chambi → compact benchmark helper
├── Chambi_benchmark_compact.json
└── output/
    ├── qwen_predictions.jsonl
    ├── doubao_predictions.jsonl
    ├── gpt_predictions.jsonl
    └── glm_predictions.jsonl
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
| **GLM** | `GLM_API_KEY`, `GLM_MODEL`; optional `GLM_BASE_URL` (Zhipu-compatible default in code — verify in console) |

Optional timeouts: `QWEN_TIMEOUT`, `OPENAI_TIMEOUT`, `GLM_TIMEOUT`, `DOUBAO_TIMEOUT` (seconds).

<a id="usage-en"></a>

### Usage

Run from the directory that contains the scripts:

```bash
export QWEN_API_KEY="sk-..." QWEN_MODEL="qwen-max"
python run_qwen_eval.py

export DOUBAO_API_KEY="..." DOUBAO_BASE_URL="https://..." DOUBAO_MODEL="..."
python run_doubao_eval.py

export OPENAI_API_KEY="sk-..." OPENAI_MODEL="gpt-4o"
python run_gpt_eval.py

export GLM_API_KEY="..." GLM_MODEL="glm-4"
python run_glm_eval.py
```

**Shared CLI**

| Flag | Meaning |
|------|---------|
| `--dataset` | Path to benchmark JSON array (default: `common_eval_utils.DEFAULT_DATASET`) |
| `--output` | JSONL path (default: `output/<provider>_predictions.jsonl`) |
| `--limit` | Max **new** items to process this run (skipped “done” ids do not count) |
| `--sleep` | Seconds to sleep after each API call (default `0.4`) |

```bash
python run_qwen_eval.py --dataset ./Chambi_benchmark_compact.json --limit 50 --sleep 0.5
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
  "error": null
}
```

- `parsed_response`: parsed object on success; `null` on failure.
- `success`: `true` only if parsing succeeded; otherwise `false` and `error` carries a message.

<a id="resume-semantics-en"></a>

### Resume semantics

An id is treated as **done** (skipped on the next run) only if **both** hold:

```python
record.get("success") is True and record.get("parsed_response") is not None
```

Rows with `success: false` are **retried**. Using a different `--output` path starts a fresh progress file.

<a id="customizing-prompt--dataset-path-en"></a>

### Customizing prompt & dataset path

- **Prompt**: edit `PROMPT_TEMPLATE` in `common_eval_utils.py`; all four runners use `build_prompt()` (`{input_text}` substitution).
- **Default dataset path**: `DEFAULT_DATASET` may point to an absolute path on the author’s machine — after cloning, switch to a **relative path** or pass `--dataset` every time.

<a id="related-tooling-en"></a>

### Related tooling

- **`generate_rot_dataset.py`**: optional pipeline from full Chambi-style inputs to compact benchmark JSON (independent of the batch inference runners).

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

- **统一评测逻辑**：`common_eval_utils.py` 负责读数据、Prompt、`extract_json_object`、JSONL 写入与已完样本集合。
- **一模型一脚本**：四套入口，默认分别写入 `output/` 下独立文件（如 `output/qwen_predictions.jsonl`）。
- **不把 gold 发给模型**：仅 `input_text` + 统一模板；`ambiguity_type`、`value_dimension`、`readings` 只留在数据里供线下对比。
- **健壮 JSON 解析**：支持 \`\`\`json 围栏与前后杂话；整块解析失败时再取首个 `{` 到最后一个 `}` 重试。
- **可靠落盘**：每条写入后 `flush()` + `os.fsync()`，适合长时间批量任务。
- **环境变量驱动**：密钥、Base URL、模型名从环境读取；可选用项目根 `.env`（不覆盖已有 `export`）。

<a id="仓库结构"></a>

### 仓库结构

```text
.
├── README.md
├── requirements.txt
├── common_eval_utils.py           # 共享逻辑
├── run_qwen_eval.py               # 通义 Qwen（DashScope 兼容）
├── run_doubao_eval.py             # 豆包 / 火山 Ark
├── run_gpt_eval.py                # OpenAI 或兼容代理
├── run_glm_eval.py                # 智谱 GLM
├── generate_rot_dataset.py        # （可选）Chambi → 精简 benchmark
├── Chambi_benchmark_compact.json  # 基准数据（JSON 数组）
└── output/                        # 默认预测输出（自动创建）
    ├── qwen_predictions.jsonl
    ├── doubao_predictions.jsonl
    ├── gpt_predictions.jsonl
    └── glm_predictions.jsonl
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
| **GLM** | `GLM_API_KEY`、`GLM_MODEL`；`GLM_BASE_URL` 有常见默认值，请以智谱控制台为准 |

可选超时：`QWEN_TIMEOUT`、`OPENAI_TIMEOUT`、`GLM_TIMEOUT`、`DOUBAO_TIMEOUT`（秒）。

<a id="运行方式"></a>

### 运行方式

在脚本所在目录执行：

```bash
export QWEN_API_KEY="sk-..." QWEN_MODEL="qwen-max"
python run_qwen_eval.py

export DOUBAO_API_KEY="..." DOUBAO_BASE_URL="https://..." DOUBAO_MODEL="..."
python run_doubao_eval.py

export OPENAI_API_KEY="sk-..." OPENAI_MODEL="gpt-4o"
python run_gpt_eval.py

export GLM_API_KEY="..." GLM_MODEL="glm-4"
python run_glm_eval.py
```

**命令行参数（四脚本一致）**

| 参数 | 说明 |
|------|------|
| `--dataset` | 基准 JSON 路径（默认见 `DEFAULT_DATASET`） |
| `--output` | JSONL 路径（默认 `output/*_predictions.jsonl`） |
| `--limit` | 本轮**最多新处理**条数（已跳过的不计） |
| `--sleep` | 每条请求后休眠秒数（默认 `0.4`） |

```bash
python run_qwen_eval.py --dataset ./Chambi_benchmark_compact.json --limit 50 --sleep 0.5
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
  "error": null
}
```

- **`parsed_response`**：解析成功为对象；失败为 `null`。
- **`success`**：成功为 `true`；否则 `false`，**`error`** 为说明字符串。

<a id="续跑规则"></a>

### 续跑规则

仅当**同时**满足下列条件时，`source_chambi_id` 视为已完成，下次运行会跳过：

```python
record.get("success") is True and record.get("parsed_response") is not None
```

**`success: false` 的样本会重跑**；更换 `--output` 即独立进度。

<a id="自定义-prompt-与数据路径"></a>

### 自定义 Prompt 与数据路径

- **Prompt**：修改 `common_eval_utils.py` 中的 `PROMPT_TEMPLATE`；`build_prompt()` 替换 `{input_text}`，四脚本共用。
- **默认数据路径**：`DEFAULT_DATASET` 可能仍是本机绝对路径；**克隆仓库后**请改为相对路径或始终使用 `--dataset`。

<a id="相关脚本"></a>

### 相关脚本

- **`generate_rot_dataset.py`**：从完整 Chambi 工作流生成/维护精简 benchmark（与批量推理脚本解耦）。

<a id="许可证"></a>

### 许可证

公开仓库前请自行添加 `LICENSE`（如 MIT）。提 Issue / PR 时请附 Python 版本、厂商、模型名及**脱敏**报错信息，便于复现。
