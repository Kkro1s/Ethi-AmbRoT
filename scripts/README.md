# Ethi-AmbRoT Scripts layout

| Directory | Purpose |
|-----------|---------|
| **`llm/`** | Batch inference: Qwen, Doubao, GPT, GLM. Phase 1: `--phase 1`. Phase 2: `--phase 2 --phase1-jsonl` 指向**本轮** Phase1 文件。默认文件名：GLM 为 `{GLM_MODEL}.jsonl`；Qwen/GPT/Doubao 为 `{provider}_{MODEL}.jsonl`。 |
| **`dataset/`** | `generate_rot_dataset.py`, `export_predictions_json.py`. |
| **`evaluation/`** | `evaluate_phase1.py`（解读恢复）、`evaluate_phase2_judge.py`（**主分数**：LLM judge，JUDGE\_* 环境变量）、`evaluate_overall.py`（端到端 overall 分数）。 |

Shared code: **`ethi_ambrot/`** — `eval_prompt.py`, `phase2_main.py`, `common_eval_utils.py`。默认 JSONL 见上表（含模型名；可用 `--output` 覆盖）。

## 生成 `output/test1` 与 `output/test2`

在仓库根目录执行。**Phase 2-main 依赖同 PROVIDER 的 Phase 1 JSONL**（scripts 会从 phase1 里筛「有效双解读」样本，再调用模型）。

### Phase 1（默认路径含模型名，见表内说明）

```bash
python scripts/llm/run_glm_eval.py --phase 1              # 续跑：跳过已存在的 sid
python scripts/llm/run_glm_eval.py --phase 1 --no-resume    # 保留原 glm.jsonl，新建 glm_1.jsonl（下一个数字递增）；Phase 2 请改 --phase1-jsonl 指向这次新文件
```

将 `run_glm_eval` 换成 `run_qwen_eval` / `run_gpt_eval` / `run_doubao_eval` 即可；可用 `--output` 指定路径。
**GLM**：`output/test1/{GLM_MODEL}.jsonl`。**Qwen / GPT / Doubao**：`output/test1/{provider}_{MODEL}.jsonl`（如 `qwen_qwen-max.jsonl`）。换模型即新文件，避免误续跑。

### Phase 2-main（`output/test2/` 下文件名规则与 Phase 1 相同）

```bash
python scripts/llm/run_glm_eval.py --phase 2 --phase1-jsonl output/test1/glm.jsonl
python scripts/llm/run_glm_eval.py --phase 2 --phase1-jsonl output/test1/glm-4-air.jsonl --no-resume
# Phase 2 的 --phase1-jsonl 须指向本轮 Phase 1 实际生成的文件。--no-resume 时在同目录递增 ``<model>_1.jsonl``…
```

Qwen / GPT / Doubao 示例：

```bash
python scripts/llm/run_qwen_eval.py --phase 2 --phase1-jsonl output/test1/qwen_qwen-max.jsonl
python scripts/llm/run_gpt_eval.py --phase 2 --phase1-jsonl output/test1/gpt_gpt-4o.jsonl
python scripts/llm/run_doubao_eval.py --phase 2 --phase1-jsonl output/test1/doubao_你的模型名.jsonl
```

**注意**：`--phase1-jsonl` 必须与当前模型一致（例如 phase2 用 GLM 时，phase1 也应是 `glm.jsonl` 或与之一一对应的解读来源）；环境变量见各 `run_*_eval.py` 文件头注释。

## Phase 1 正式测评（`evaluate_phase1.py`）

在 **已有 Phase 1 JSONL**（如 `output/test1/glm.jsonl`）的前提下，对 gold benchmark 做严格解读恢复评测。

```bash
python scripts/evaluation/evaluate_phase1.py \
  --predictions output/test1/glm.jsonl \
  --output-dir output/phase1_eval/glm
```

输出：
- `phase1_eval_summary.json`
- `phase1_eval_details.jsonl`

`summary.metrics` 里建议重点看：
- **`two_reading_recovery_rate`**：Phase 1 主指标。要求预测的两条 reading 与 gold 的两条 reading 进行**严格一对一最优匹配**；同一条预测不能同时覆盖两个 gold reading。
- **`two_valid_distinct_reading_rate`**：辅助指标。只表示模型是否产出了两个有效且彼此不同的 reading，不代表它们已经与 gold 对齐成功。
- **`reading_coverage_recall`**：辅助指标。表示 gold reading 被预测覆盖到的比例。

兼容字段 **`double_correct_rate`** 与 `two_reading_recovery_rate` 含义相同。

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

默认 **`--resume`**：跳过 detail 中已成功 judge 的 `source_ethi_ambrot_id`；`--no-resume` 会对队列中每一项都再次请求 judge（可能产生重复行，summary 按 sid 取最后一行汇总）。`--limit N` 用于调试。

主分数见 summary 的 **`metrics`**。当前口径下：
- 评测前会先按 reading 文本对预测 A/B 与 gold A/B 做**严格两种配对中的最优对齐**，避免仅因 A/B 顺序互换而误扣分。
- `avg_item_score` / `avg_score_rot_overall` 可作为 Phase 2 主结果。
- **`metrics_supplementary`** 为辅助分析：除字符串层 primary / 价值集合命中率外，也包含 judge 返回的 `norm_match` / `obligation_match` / `advice_match` / `value_match` 一致率，**非主结论**。

## Overall End-to-End 测评（`evaluate_overall.py`）

在 **已有 Phase 1 和 Phase 2 评测结果** 的前提下，计算 benchmark 的 end-to-end overall 分数。

```bash
python scripts/evaluation/evaluate_overall.py \
  --phase1-details output/phase1_eval/phase1_eval_details.jsonl \
  --phase2-details output/phase2_judge_eval/phase2_judge_detail.jsonl \
  --phase1-summary output/phase1_eval/phase1_eval_summary.json \
  --phase2-summary output/phase2_judge_eval/phase2_judge_summary.json \
  --output-dir output/overall_eval
```

输出：
- `overall_summary.json`
- `overall_details.jsonl`

`summary.metrics` 里的主指标：
- **`overall_e2e_score`**：benchmark 的 end-to-end 总分（0-1 scale）。
- **`overall_e2e_score_percent`**：百分制总分。
- **`phase1_gate_rate`**：Phase 1 严格双解读恢复通过率。
- **`phase2_conditional_success_rate`**：在 Phase 1 通过的样本中，Phase 2 judge 成功率。

计分逻辑：
- Phase 1 未严格恢复双解读 → 该样本得 0 分
- Phase 2 未成功 judge → 该样本得 0 分
- 两阶段都成功 → 该样本得分 = `(phase2_item_score - 1) / 2`（将 1-3 映射到 0-1）
