# Scripts layout

| Directory | Purpose |
|-----------|---------|
| **`llm/`** | Batch inference: Qwen, Doubao, GPT, GLM. Phase 1: `--phase 1`. Phase 2-main: `--phase 2 --phase1-jsonl output/test1/<provider>.jsonl`. |
| **`dataset/`** | `generate_rot_dataset.py`, `export_predictions_json.py`. |
| **`evaluation/`** | `evaluate_phase1.py`（解读恢复）、`evaluate_phase2_main.py`（启发式自动指标）、`evaluate_phase2_judge.py`（**主分数**：LLM judge，JUDGE\_\* 环境变量）。 |

Shared code: **`ethi_ambrot/`** — `eval_prompt.py` (`PROMPT_TEST1_TEMPLATE`, `PROMPT_PHASE2_MAIN_TEMPLATE`), `phase2_main.py` (filter + parse), `common_eval_utils.py`. Default JSONL: **`output/test1/<provider>.jsonl`**, **`output/test2/<provider>.jsonl`**.

## 生成 `output/test1` 与 `output/test2`

在仓库根目录执行。**Phase 2-main 依赖同 PROVIDER 的 Phase 1 JSONL**（scripts 会从 phase1 里筛「有效双解读」样本，再调用模型）。

### Phase 1（写入 `output/test1/<provider>.jsonl`）

```bash
python scripts/llm/run_glm_eval.py --phase 1              # 续跑：跳过已存在的 sid
python scripts/llm/run_glm_eval.py --phase 1 --no-resume    # 删默认输出后整表重跑
```

将 `run_glm_eval` 换成 `run_qwen_eval` / `run_gpt_eval` / `run_doubao_eval` 即可；可用 `--output` 指定路径。

### Phase 2-main（写入 `output/test2/<provider>.jsonl`）

```bash
python scripts/llm/run_glm_eval.py --phase 2 --phase1-jsonl output/test1/glm.jsonl
python scripts/llm/run_glm_eval.py --phase 2 --phase1-jsonl output/test1/glm.jsonl --no-resume
```

Qwen / GPT / Doubao 示例：

```bash
python scripts/llm/run_qwen_eval.py --phase 2 --phase1-jsonl output/test1/qwen.jsonl
python scripts/llm/run_gpt_eval.py --phase 2 --phase1-jsonl output/test1/gpt.jsonl
python scripts/llm/run_doubao_eval.py --phase 2 --phase1-jsonl output/test1/doubao.jsonl
```

**注意**：`--phase1-jsonl` 必须与当前模型一致（例如 phase2 用 GLM 时，phase1 也应是 `glm.jsonl` 或与之一一对应的解读来源）；环境变量见各 `run_*_eval.py` 文件头注释。

## Phase 2 LLM-as-judge 测评（`evaluate_phase2_judge.py`）

在 **已有 Phase 2 JSONL**（如 `output/test2/glm.jsonl`）的前提下，用独立 judge 模型打分（OpenAI 兼容接口）。`.env` 中配置：

- **`JUDGE_API_KEY`**（必填）
- **`JUDGE_BASE_URL`**（必填，可去掉尾部 `/chat/completions`）
- **`JUDGE_MODEL`**（必填）
- **`JUDGE_TIMEOUT`**（可选，秒，默认 180）

```bash
python scripts/evaluation/evaluate_phase2_judge.py \
  --predictions output/test2/glm.jsonl \
  --detail-output output/phase2_judge_eval/glm_detail.jsonl \
  --summary-output output/phase2_judge_eval/glm_summary.json
```

默认 **`--resume`**：跳过 detail 中已成功 judge 的 `source_chambi_id`；`--no-resume` 会对队列中每一项都再次请求 judge（可能产生重复行，summary 按 sid 取最后一行汇总）。`--limit N` 用于调试。

主分数见 summary 的 **`metrics`**；**`metrics_supplementary`** 仅为字符串层 primary / 价值集合命中率，**非主结论**。
