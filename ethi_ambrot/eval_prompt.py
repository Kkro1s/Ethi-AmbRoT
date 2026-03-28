"""
多模型两阶段评测共用的用户提示（测试1：解读；Phase 2-main：给定双解读下的 RoT/价值）。

- ``PROMPT_TEST1_TEMPLATE`` / ``build_prompt_test1``
- ``PROMPT_PHASE2_MAIN_TEMPLATE`` / ``build_prompt_phase2_main``（不向模型泄露 gold）

过滤与解析：``ethi_ambrot.phase2_main``。
"""

from __future__ import annotations

INPUT_PLACEHOLDER = "{input_text}"
READING_A_PLACEHOLDER = "{reading_a}"
READING_B_PLACEHOLDER = "{reading_b}"

PROMPT_TEST1_TEMPLATE = """请解读下面这句话：

「{input_text}」

请只完成以下任务：
1. 先写出原句（照抄输入句即可）；
2. 给出你认为最自然的一种主解读（解读A）；
3. 若你认为还有明显不同的理解，再写解读B；若没有第二种合理解读，可省略「解读B」这一行或写「解读B：无」。

不要分析价值、道德、社会规范或建议。

请严格按下面版式回答（每行以标签开头）：

原句子：……
解读A：……
解读B：……"""


PROMPT_PHASE2_MAIN_TEMPLATE = """你现在是一名谨慎的中文规范与价值分析助手。

下面给出「原始句子」以及已经确定的两种语义解读（解读A、解读B）。请**不要**再讨论这句话是否有歧义；把两种解读都当作已知前提。

原始句子：
「{input_text}」

解读A（已知）：
{reading_a}

解读B（已知）：
{reading_b}

请**分别**针对解读A与解读B，各写一篇简短分析。价值维度字段只能从以下四个英文选项中择一填写（主要、次级若无可填 None 或 无）：
Family, Mianzi, Harmony, Public Morality

请**严格**按如下半结构化版式输出（标签一字不差；先完整写完【解读A】整块，再写【解读B】）：

【解读A】
社会规范：……
伦理义务：……
建议：……
主要价值维度：……
次级价值维度：……
理由：……

【解读B】
社会规范：……
伦理义务：……
建议：……
主要价值维度：……
次级价值维度：……
理由：……"""


def build_prompt_test1(input_text: str) -> str:
    """测试1：原句 + 解读A（必填）+ 可选解读B；不涉及价值或 RoT。"""
    return PROMPT_TEST1_TEMPLATE.replace(INPUT_PLACEHOLDER, input_text)


def build_prompt_phase2_main(input_text: str, reading_a: str, reading_b: str) -> str:
    """Phase 2-main：仅基于给定双解读做 RoT 与价值分析。"""
    return (
        PROMPT_PHASE2_MAIN_TEMPLATE.replace(INPUT_PLACEHOLDER, input_text.strip())
        .replace(READING_A_PLACEHOLDER, reading_a.strip())
        .replace(READING_B_PLACEHOLDER, reading_b.strip())
    )


build_prompt_test2 = build_prompt_phase2_main
