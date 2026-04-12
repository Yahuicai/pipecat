"""Generate model reference answers for interview questions.

This module provides LLM-powered generation of reference answer transcripts
for each practice question, producing the kind of structured, well-organized
response a candidate should aim for.
"""

import asyncio
import os
from dataclasses import dataclass

import openai
from loguru import logger

from interview_practice import PracticeQuestion

_SYSTEM_PROMPT = (
    "你是一名资深广州事业单位面试培训师。"
    "请针对以下面试题目，给出一份高质量的参考逐字稿答案。\n\n"
    "要求：\n"
    "1. 答案需要像真实面试中考生口述一样自然流畅，适合直接朗读。\n"
    "2. 结构清晰：先表明态度/总论点，再分层论述（2-3个层次），最后简短总结。\n"
    "3. 体现公务员思维：政治站位、群众意识、务实态度。\n"
    "4. 控制在 300-400 字，约 2-3 分钟口述时长。\n"
    "5. 不要使用 emoji、项目符号或 markdown 格式，纯文字输出。\n"
)


@dataclass
class ReferenceAnswer:
    question_index: int
    question_prompt: str
    category: str
    answer: str


async def generate_reference_answer(
    question: PracticeQuestion,
    index: int,
) -> ReferenceAnswer:
    """Generate a single reference answer for a question via streaming LLM."""
    prompt = f"【{question.category}】{question.prompt}"
    try:
        client = openai.AsyncOpenAI(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("ARK_BASE_URL"),
        )
        resp = await client.chat.completions.create(
            model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 4000}},
        )
        answer_text = resp.choices[0].message.content or "参考答案生成失败"
    except Exception as e:
        logger.error(f"Reference answer generation failed for Q{index + 1}: {e}")
        answer_text = f"参考答案生成失败：{e}"

    return ReferenceAnswer(
        question_index=index,
        question_prompt=question.prompt,
        category=question.category,
        answer=answer_text,
    )


async def generate_all_reference_answers(
    questions: list[PracticeQuestion],
) -> list[ReferenceAnswer]:
    """Generate reference answers for all questions concurrently."""
    tasks = [generate_reference_answer(q, i) for i, q in enumerate(questions)]
    return await asyncio.gather(*tasks)
