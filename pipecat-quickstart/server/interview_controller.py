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
    BotStoppedSpeakingFrame,
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
from reference_answer import generate_reference_answer


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
        self._waiting_for_tts_done: bool = False
        # Pre-fetched reference answers: index -> answer text
        self._ref_answers: dict[int, str] = {}
        self._ref_tasks: dict[int, object] = {}
        # Per-question LLM comments: index -> comment text
        self._q_comments: dict[int, str] = {}
        self._q_comment_tasks: dict[int, object] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStoppedSpeakingFrame) and self._waiting_for_tts_done:
                self._waiting_for_tts_done = False
                self._state = _State.WAITING_ANSWER
                self._current_transcript_parts.clear()
                logger.info("TTS finished, starting answer timer")
                await self._send_ui({"type": "waiting_answer"})
            await self.push_frame(frame, direction)
            return

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
            elapsed = int((data.get("d") or {}).get("elapsed_secs", 0))
            await self._on_done_answering(elapsed)
            return True
        return False

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _on_start_practice(self):
        if self._state not in (_State.IDLE, _State.FINISHED):
            return
        logger.info("Starting practice session")
        # Reset all per-session state
        self._current_transcript_parts.clear()
        self._waiting_for_tts_done = False
        self._ref_answers.clear()
        self._ref_tasks.clear()
        self._q_comments.clear()
        self._q_comment_tasks.clear()
        self._session = build_practice_session()
        self._state = _State.READING_QUESTION
        # Pre-fetch reference answers for all questions in background
        for i, q in enumerate(self._session.questions):
            self._ref_tasks[i] = self.create_task(self._fetch_ref_answer(q, i), f"ref_answer_{i}")
        await self._read_current_question(with_intro=True)

    async def _on_done_answering(self, elapsed_secs: int = 0):
        if self._state != _State.WAITING_ANSWER or self._session is None:
            return

        answer = " ".join(self._current_transcript_parts).strip() or "（未检测到回答）"
        q_index = self._session.current_question_index
        self._session.record_answer(answer, elapsed_secs)
        self._current_transcript_parts.clear()
        logger.info(f"Answer recorded for Q{q_index}: {answer[:80]}...")

        await self._send_ui(
            {
                "type": "answer_transcript",
                "index": q_index + 1,
                "text": answer,
            }
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

        self._waiting_for_tts_done = True
        await self.push_frame(TTSSpeakFrame(text=tts_text))

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
    # Pre-fetch reference answer (background, started at question draw)
    # ------------------------------------------------------------------

    async def _fetch_ref_answer(self, q, index: int):
        """Fetch reference answer in background; stored for delivery in final report."""
        try:
            ref = await generate_reference_answer(q, index)
            self._ref_answers[index] = ref.answer
            logger.info(f"Reference answer ready for Q{index + 1}")
        except Exception as e:
            logger.error(f"Reference answer generation failed for Q{index + 1}: {e}")

    # ------------------------------------------------------------------
    # Per-question comment (started after all answers collected)
    # ------------------------------------------------------------------

    async def _generate_question_comment(self, q, answer: str, index: int):
        """Call LLM to generate a per-question comment for one question."""
        prompt = (
            f"你是一名广州事业单位面试陪练教练。"
            f"请对以下一道面试题和考生的回答给出简短点评。\n\n"
            f"题目【{q.category}】：{q.prompt}\n\n"
            f"考生回答：{answer}\n\n"
            "输出要求：\n"
            "1. 点评内容包括答题表现、优点和不足。\n"
            "2. 控制在80-120字以内，语气直接、具体。\n"
            "3. 纯文字输出，不要使用 emoji 或格式符号。\n"
        )
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("ARK_API_KEY"),
                base_url=os.getenv("ARK_BASE_URL"),
            )
            resp = await client.chat.completions.create(
                model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "disabled"}},
            )
            comment = (resp.choices[0].message.content or "").strip()
            self._q_comments[index] = comment
            logger.info(f"Per-question comment ready for Q{index + 1}")
        except Exception as e:
            logger.error(f"Per-question comment failed for Q{index + 1}: {e}")
            self._q_comments[index] = ""

    # ------------------------------------------------------------------
    # Final feedback + report assembly
    # ------------------------------------------------------------------

    async def _generate_and_deliver_feedback(self):
        import asyncio

        # Launch per-question comment tasks concurrently (answers now available)
        for i, q in enumerate(self._session.questions):
            answer = self._session.answers[i] if i < len(self._session.answers) else ""
            self._q_comment_tasks[i] = self.create_task(
                self._generate_question_comment(q, answer, i), f"q_comment_{i}"
            )

        # Stream overall summary to frontend while background tasks run
        history = self._session.formatted_history()
        prompt = (
            "你是一名广州事业单位面试陪练教练。"
            "请基于下面三道题和考生回答，给出总体评价和改进建议。\n\n"
            f"{history}\n\n"
            "输出要求：\n"
            "1. 只输出两部分，每部分前用方括号标记。\n"
            "2. [总体评价]：概括三道题的整体表现，约100字。\n"
            "3. [改进建议]：给出2-3条具体可操作的建议，约100字。\n"
            "4. 语气像个人陪练教练，直接、具体，不要空泛。\n"
            "5. 不要使用 emoji、项目符号或 markdown 格式。\n"
        )

        feedback_parts: list[str] = []
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("ARK_API_KEY"),
                base_url=os.getenv("ARK_BASE_URL"),
            )
            stream = await client.chat.completions.create(
                model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
                messages=[{"role": "user", "content": prompt}],
                extra_body={"thinking": {"type": "disabled"}},
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    feedback_parts.append(delta)
                    await self._send_ui({"type": "feedback_chunk", "text": delta})
        except Exception as e:
            logger.error(f"LLM feedback call failed: {e}")
            feedback_parts.append(f"抱歉，总点评生成失败：{e}")
            await self._send_ui({"type": "feedback_chunk", "text": feedback_parts[-1]})

        feedback_text = "".join(feedback_parts)
        await self._send_ui({"type": "feedback_done"})

        # Parse overall + recommendations from streamed text
        overall, recommendations = self._parse_summary(feedback_text)

        # Wait for all background tasks (ref answers + per-question comments)
        all_bg_tasks = list(self._ref_tasks.values()) + list(self._q_comment_tasks.values())
        pending = [t for t in all_bg_tasks if isinstance(t, asyncio.Task) and not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Assemble structured report
        n = len(self._session.questions)
        self._session.reference_answers = [self._ref_answers.get(i, "") for i in range(n)]
        self._session.question_comments = [self._q_comments.get(i, "") for i in range(n)]

        report_questions = []
        for i, q in enumerate(self._session.questions):
            elapsed = self._session.answer_times[i] if i < len(self._session.answer_times) else 0
            mm, ss = divmod(elapsed, 60)
            report_questions.append(
                {
                    "index": i + 1,
                    "category": q.category,
                    "prompt": q.prompt,
                    "answer": self._session.answers[i] if i < len(self._session.answers) else "",
                    "elapsed_secs": elapsed,
                    "elapsed_label": f"{mm}分{ss:02d}秒",
                    "reference_answer": self._ref_answers.get(i, ""),
                    "comment": self._q_comments.get(i, ""),
                }
            )

        await self._send_ui(
            {
                "type": "report",
                "questions": report_questions,
                "overall": overall,
                "recommendations": recommendations,
            }
        )

        self._session.final_feedback = feedback_text
        self._session.final_feedback_requested = True

        record_path = save_practice_record(self._session)
        logger.info(f"Practice record saved to {record_path}")

        self._state = _State.FINISHED
        logger.info("Interview practice finished")
        await self._send_ui({"type": "session_ready"})

    @staticmethod
    def _parse_summary(text: str) -> tuple[str, str]:
        """Extract [总体评价] and [改进建议] sections from streamed feedback text."""
        import re

        parts = re.split(r"\[([^\]]+)\]", text)
        sections: dict[str, str] = {}
        for i in range(1, len(parts) - 1, 2):
            sections[parts[i].strip()] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        overall = sections.get("总体评价", text.strip())
        recommendations = sections.get("改进建议", "")
        return overall, recommendations

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
