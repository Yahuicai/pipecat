"""Interview practice controller -- event-driven state machine.

Sits in the pipeline between STT and TTS.  Accumulates transcripts,
reacts to RTVI client-messages (start_practice / done_answering), reads
questions via TTSSpeakFrame, and calls the LLM once for the final summary.
"""

import os
from enum import Enum
from typing import Any

import openai
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from interview_practice import (
    PracticeSession,
    build_practice_session,
    save_practice_record,
)


class _State(Enum):
    IDLE = "idle"
    READING_QUESTION = "reading_question"
    WAITING_ANSWER = "waiting_answer"
    GENERATING_FEEDBACK = "generating_feedback"
    FINISHED = "finished"


class InterviewController(FrameProcessor):
    """Event-driven interview controller.

    State transitions:
        idle -> reading_question   (on start_practice)
        reading_question -> waiting_answer  (immediately after TTS push)
        waiting_answer -> reading_question  (on done_answering, questions left)
        waiting_answer -> generating_feedback (on done_answering, last question)
        generating_feedback -> finished     (after feedback delivered)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = _State.IDLE
        self._session: PracticeSession | None = None
        self._current_transcript_parts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TranscriptionFrame):
                await self._handle_transcription(frame)
                await self.push_frame(frame, direction)
                return

            if isinstance(frame, InputTransportMessageFrame):
                handled = await self._handle_transport_message(frame.message)
                if handled:
                    return

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Transport messages (RTVI client-message)
    # ------------------------------------------------------------------

    async def _handle_transport_message(self, message: Any) -> bool:
        """Return True if this message was consumed (don't forward)."""
        if not isinstance(message, dict):
            return False
        label = message.get("label")
        msg_type = message.get("type")

        if label != "rtvi-ai" or msg_type != "client-message":
            return False

        data = message.get("data", {})
        action = data.get("t", "")

        if action == "start_practice":
            await self._on_start_practice()
            return True
        if action == "done_answering":
            await self._on_done_answering()
            return True
        return False

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _on_start_practice(self):
        if self._state != _State.IDLE:
            return
        logger.info("Starting practice session")
        self._session = build_practice_session()
        self._state = _State.READING_QUESTION
        await self._read_current_question(with_intro=True)

    async def _on_done_answering(self):
        if self._state != _State.WAITING_ANSWER or self._session is None:
            return

        answer = " ".join(self._current_transcript_parts).strip() or "（未检测到回答）"
        self._session.record_answer(answer)
        self._current_transcript_parts.clear()
        logger.info(
            f"Answer recorded for Q{self._session.current_question_index}: {answer[:80]}..."
        )

        if self._session.is_complete:
            self._state = _State.GENERATING_FEEDBACK
            await self._send_ui({"type": "generating_feedback"})
            await self._send_ui(
                {
                    "type": "progress",
                    "current": self._session.total_questions,
                    "total": self._session.total_questions,
                    "label": "生成总点评中...",
                }
            )
            await self._generate_and_deliver_feedback()
        else:
            self._state = _State.READING_QUESTION
            await self._read_current_question(with_intro=False)

    # ------------------------------------------------------------------
    # Question reading (fixed text -> TTS, no LLM)
    # ------------------------------------------------------------------

    async def _read_current_question(self, *, with_intro: bool):
        idx = self._session.current_question_index
        q = self._session.questions[idx]

        await self._send_ui(
            {
                "type": "progress",
                "current": idx + 1,
                "total": self._session.total_questions,
            }
        )
        await self._send_ui(
            {
                "type": "question",
                "index": idx + 1,
                "category": q.category,
                "text": q.prompt,
            }
        )

        if with_intro:
            tts_text = f"好的，今天我们练习三道题。第一题，{q.category}类。{q.prompt}"
        else:
            tts_text = f"收到，我们进入下一题。第{idx + 1}题，{q.category}类。{q.prompt}"

        await self.push_frame(TTSSpeakFrame(text=tts_text))
        self._state = _State.WAITING_ANSWER
        self._current_transcript_parts.clear()
        await self._send_ui({"type": "waiting_answer"})

    # ------------------------------------------------------------------
    # Transcript accumulation
    # ------------------------------------------------------------------

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if self._state == _State.WAITING_ANSWER and frame.text.strip():
            self._current_transcript_parts.append(frame.text.strip())
            await self._send_ui(
                {
                    "type": "transcript",
                    "text": " ".join(self._current_transcript_parts),
                }
            )

    # ------------------------------------------------------------------
    # Final feedback (one-shot LLM call outside pipeline)
    # ------------------------------------------------------------------

    async def _generate_and_deliver_feedback(self):
        history = self._session.formatted_history()
        prompt = (
            "你是一名广州事业单位面试陪练教练。"
            "请基于下面三道题和考生回答，给出一次统一总点评。\n\n"
            f"{history}\n\n"
            "输出要求：\n"
            "1. 只输出总点评。\n"
            "2. 总点评按四部分组织：总体表现、亮点、主要问题、改进建议。\n"
            "3. 语气像个人陪练教练，直接、具体，不要空泛。\n"
            "4. 适合语音播报，控制在大约300字以内。\n"
        )

        try:
            client = openai.OpenAI(
                api_key=os.getenv("ARK_API_KEY"),
                base_url=os.getenv("ARK_BASE_URL"),
            )
            resp = client.chat.completions.create(
                model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "disabled"}},
            )
            feedback = resp.choices[0].message.content or "未能生成点评"
        except Exception as e:
            logger.error(f"LLM feedback call failed: {e}")
            feedback = f"抱歉，总点评生成失败：{e}"

        self._session.final_feedback = feedback
        self._session.final_feedback_requested = True

        record_path = save_practice_record(self._session)
        logger.info(f"Practice record saved to {record_path}")

        await self._send_ui({"type": "feedback", "text": feedback})
        await self.push_frame(TTSSpeakFrame(text=feedback))

        self._state = _State.FINISHED
        logger.info("Interview practice finished")

    # ------------------------------------------------------------------
    # Helper: send JSON to the frontend via data channel
    # ------------------------------------------------------------------

    async def _send_ui(self, data: dict):
        msg = {
            "label": "rtvi-ai",
            "type": "server-message",
            "data": data,
        }
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=msg),
            FrameDirection.DOWNSTREAM,
        )
