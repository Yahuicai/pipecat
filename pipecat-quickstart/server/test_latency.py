"""
Doubao 模型思考模式延迟对比测试

测试 doubao-seed-2-0-pro 在 thinking enabled/disabled 两种模式下的：
- TTFB (Time To First token)
- 总完成时间
- token 用量（含 reasoning tokens）

运行：
    uv run test_latency.py
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

CLIENT = AsyncOpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
)

# 使用推理模型测试思考开关
MODEL = "doubao-seed-2-0-pro-260215"

PROMPT = "用一两句话介绍一下你自己。"


async def benchmark(label: str, thinking_type: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"测试: {label}  (thinking={thinking_type})")
    print(f"{'=' * 50}")

    t_start = time.perf_counter()
    t_first_token = None
    full_text = ""
    usage = None

    stream = await CLIENT.chat.completions.create(
        model=MODEL,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": PROMPT}],
        extra_body={
            "thinking": {
                "type": thinking_type,
            }
        },
    )

    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
            continue

        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if t_first_token is None:
                t_first_token = time.perf_counter()
                ttfb = t_first_token - t_start
                print(f"  TTFB:        {ttfb:.3f}s")
            full_text += delta

    t_end = time.perf_counter()
    total = t_end - t_start

    print(f"  总耗时:      {total:.3f}s")
    print(f"  回复内容:    {full_text.strip()}")
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", "?")
        completion_tokens = getattr(usage, "completion_tokens", "?")
        # reasoning_tokens 在 completion_tokens_details 里
        reasoning_tokens = "?"
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", "?")
        print(
            f"  Token 用量:  prompt={prompt_tokens}, completion={completion_tokens}, reasoning={reasoning_tokens}"
        )


async def main():
    print(f"模型: {MODEL}")
    print(f"Prompt: {PROMPT}")

    # 先测试 disabled，再测试 enabled，各跑两次取稳定值
    for i in range(1, 3):
        await benchmark(f"第{i}次 - 关闭思考", thinking_type="disabled")

    for i in range(1, 3):
        await benchmark(f"第{i}次 - 开启思考", thinking_type="enabled")


if __name__ == "__main__":
    asyncio.run(main())
