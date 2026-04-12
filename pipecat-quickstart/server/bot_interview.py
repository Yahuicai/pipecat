"""Interview practice bot -- button-controlled, no VAD auto-trigger.

Pipeline:  Transport.input() -> STT -> InterviewController -> TTS -> Transport.output()

The InterviewController handles all state, reads questions via TTSSpeakFrame
(no LLM in the main pipeline), and only calls LLM once for the final summary.
"""

import json
import os
import ssl
from typing import AsyncGenerator, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.tracing.service_decorators import traced_tts

from interview_controller import InterviewController

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# MiniMax WebSocket TTS (same as bot.py -- extracted here for reuse)
# ---------------------------------------------------------------------------


class MiniMaxWSTTSService(TTSService):
    """MiniMax TTS via WebSocket streaming API."""

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
                self._ws_url, additional_headers=headers, ssl=self._ssl_ctx
            ) as ws:
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
                await ws.send(json.dumps({"event": "task_continue", "text": text}))
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


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport):
    """Assemble and run the interview practice pipeline."""
    logger.info("Starting interview practice bot")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language=Language.ZH,
        ),
    )

    tts = MiniMaxWSTTSService(
        api_key=os.getenv("MINIMAX_API_KEY", ""),
        group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        voice="female-shaonv",
    )

    controller = InterviewController()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            controller,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ---------------------------------------------------------------------------
# Entry point (called by main.py via runner_args)
# ---------------------------------------------------------------------------


async def bot(runner_args: RunnerArguments):
    """Bot entry point -- only supports SmallWebRTC."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
        return

    connection: SmallWebRTCConnection = runner_args.webrtc_connection
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )
    await run_bot(transport)
