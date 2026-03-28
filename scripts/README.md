# Scripts layout

| Directory | Purpose |
|-----------|---------|
| **`llm/`** | Batch inference: Qwen, Doubao, GPT, GLM. Phase 1: `--phase 1`. Phase 2-main: `--phase 2 --phase1-jsonl output/test1/<provider>.jsonl`. |
| **`dataset/`** | `generate_rot_dataset.py`, `export_predictions_json.py`. |
| **`evaluation/`** | `evaluate_phase2_main.py` — metrics vs gold (RoT, value, differential A/B). |

Shared code: **`ethi_ambrot/`** — `eval_prompt.py` (`PROMPT_TEST1_TEMPLATE`, `PROMPT_PHASE2_MAIN_TEMPLATE`), `phase2_main.py` (filter + parse), `common_eval_utils.py`. Default JSONL: **`output/test1/<provider>.jsonl`**, **`output/test2/<provider>.jsonl`**.
