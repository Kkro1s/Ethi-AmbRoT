"""
多模型两阶段评测共用的用户提示（测试1：歧义与解读；测试2：占位 RoT/价值）。

- 测试1：``build_prompt_test1`` — 仅语言理解，不涉及价值与 RoT。
- 测试2：``build_prompt_test2`` — 给定两条解读后的分析任务；正式模板可待定稿后只改本文件。

修改任务说明时，只改 ``PROMPT_TEST1_TEMPLATE`` / ``PROMPT_TEST2_TEMPLATE`` 即可。
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


PROMPT_TEST2_TEMPLATE = """

下面的输入句在语义上可能存在两种理解，请先阅读「原始句子」与「解读A」「解读B」（每条解读已是尽量消歧的表述）。

原始句子：
「{input_text}」

解读A：
{reading_a}

解读B：
{reading_b}

请分别针对解读A与解读B完成分析（本题为测试2占位说明，正式字段与输出格式待定稿）：
- 各自可能激活的社会规范、伦理义务、可行的行动建议；
- 各自与常见中国价值维度（如家庭、面子、和谐、公德等）的对应关系。

请用清晰的中文分条回答，区分「针对解读A」与「针对解读B」两部分；暂不要求 JSON。"""


def build_prompt_test1(input_text: str) -> str:
    """测试1：原句 + 解读A（必填）+ 可选解读B；不涉及价值或 RoT。"""
    return PROMPT_TEST1_TEMPLATE.replace(INPUT_PLACEHOLDER, input_text)


def build_prompt_test2(input_text: str, reading_a: str, reading_b: str) -> str:
    """测试2：在固定两条解读上做规范/价值推理（占位模板；勿传入 gold_rot 等标签）。"""
    return (
        PROMPT_TEST2_TEMPLATE.replace(INPUT_PLACEHOLDER, input_text)
        .replace(READING_A_PLACEHOLDER, reading_a.strip())
        .replace(READING_B_PLACEHOLDER, reading_b.strip())
    )
