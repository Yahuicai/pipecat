#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Pipecat Voice Agent

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech

Required AI services:
- Deepgram (Speech-to-Text, Chinese)
- ARK / Doubao pro (LLM, OpenAI-compatible, non-reasoning)
- MiniMax (Text-to-Speech, Chinese, via WebSocket)

Run the bot using::

    uv run bot.py
"""

import json
import os
import ssl
from typing import AsyncGenerator, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, LLMRunFrame, StartFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.tracing.service_decorators import traced_tts

load_dotenv(override=True)


class MiniMaxWSTTSService(TTSService):
    """MiniMax TTS via WebSocket API (wss://api.minimaxi.com/ws/v1/t2a_v2).

    使用 WebSocket 流式接口，延迟低，适合国内访问。
    """

    def __init__(
        self,
        *,
        api_key: str,
        group_id: str,
        model: str = "speech-02-turbo",
        voice: str = "female-shaonv",
        ws_url: str = "wss://api.minimaxi.com/ws/v1/t2a_v2",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )
        self._api_key = api_key
        self._group_id = group_id
        self._model = model
        self._voice = voice
        self._ws_url = ws_url
        self._ssl_ctx = ssl.create_default_context()

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        logger.debug(
            f"MiniMaxWSTTS initialized: sample_rate={self.sample_rate}, voice={self._voice}"
        )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS via WebSocket [{text}]")

        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with websockets.connect(
                self._ws_url,
                additional_headers=headers,
                ssl=self._ssl_ctx,
            ) as ws:
                # task_start
                await ws.send(
                    json.dumps(
                        {
                            "event": "task_start",
                            "model": self._model,
                            "voice_setting": {
                                "voice_id": self._voice,
                                "speed": 1.0,
                                "vol": 1.0,
                                "pitch": 0,
                            },
                            "audio_setting": {
                                "format": "pcm",
                                "sample_rate": self.sample_rate,
                                "bitrate": 128000,
                                "channel": 1,
                            },
                        }
                    )
                )

                # task_continue (send text)
                await ws.send(
                    json.dumps(
                        {
                            "event": "task_continue",
                            "text": text,
                        }
                    )
                )

                # task_finish
                await ws.send(json.dumps({"event": "task_finish"}))

                await self.start_tts_usage_metrics(text)

                async for message in ws:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"{self}: JSON decode error: {e}")
                        continue

                    event = data.get("event", "")

                    if event == "task_failed":
                        err = data.get("base_resp", {}).get("status_msg", "unknown error")
                        logger.error(f"{self}: TTS task failed: {err}")
                        yield ErrorFrame(error=f"MiniMax TTS error: {err}")
                        return

                    if "data" in data:
                        audio_hex = data["data"].get("audio", "")
                        is_final = data["data"].get("is_final", False)

                        if audio_hex:
                            try:
                                audio_bytes = bytes.fromhex(audio_hex)
                                await self.stop_ttfb_metrics()
                                yield TTSAudioRawFrame(
                                    audio=audio_bytes,
                                    sample_rate=self.sample_rate,
                                    num_channels=1,
                                    context_id=context_id,
                                )
                            except ValueError as e:
                                logger.error(f"{self}: hex decode error: {e}")

                        if is_final:
                            break

                    elif event == "task_finished":
                        break

        except Exception as e:
            logger.error(f"{self}: WebSocket error: {e}")
            yield ErrorFrame(error=f"MiniMax WS TTS error: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # Speech-to-Text service - 中文识别
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language=Language.ZH,
        ),
    )

    # Text-to-Speech service - MiniMax WebSocket，支持中文，国内可访问
    tts = MiniMaxWSTTSService(
        api_key=os.getenv("MINIMAX_API_KEY", ""),
        group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        voice="female-shaonv",  # 可选: male-qn-qingse / female-chengshu
    )

    # LLM service - 通过 extra_body 关闭推理模型的思考模式，消除 reasoning tokens 延迟
    llm = OpenAILLMService(
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        settings=OpenAILLMService.Settings(
            model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
            system_instruction=(
                "你是一个语音对话助手。你的回复会被直接朗读，"
                "请避免使用 emoji、列表符号或其他无法朗读的格式。"
                "用简洁、自然的中文口语回答。"
            ),
            extra={
                "extra_body": {
                    "thinking": {"type": "disabled"},
                }
            },
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,  # 默认 0.2，适当增大减少误截断
                    confidence=0.6,  # 默认 0.7，稍低避免漏检
                )
            ),
        ),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message({"role": "user", "content": "请用中文简单介绍一下你自己。"})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport = None

    match runner_args:
        case DailyRunnerArguments():
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "Pipecat Bot",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
