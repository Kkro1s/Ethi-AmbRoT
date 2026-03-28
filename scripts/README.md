# Scripts layout

| Directory | Purpose |
|-----------|---------|
| **`llm/`** | Batch inference for the benchmark: Qwen, Doubao, GPT, GLM (OpenAI-compatible clients). Run from repo root, e.g. `python scripts/llm/run_qwen_eval.py`. |
| **`dataset/`** | Data tooling: `generate_rot_dataset.py` (Chambi → compact JSON), `export_predictions_json.py` (prediction JSONL → single JSON). |

Shared library code lives in **`ethi_ambrot/`** at the repo root. User messages are defined in **`ethi_ambrot/eval_prompt.py`** (`PROMPT_TEST1_TEMPLATE`, `PROMPT_TEST2_TEMPLATE`); all runners use **`--phase 1|2`** with defaults **`output/test1/<provider>.jsonl`** and **`output/test2/<provider>.jsonl`**.
