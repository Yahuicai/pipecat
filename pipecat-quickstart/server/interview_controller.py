"""Interview practice controller -- event-driven state machine.

Sits in the pipeline between STT and TTS. Displays all 3 questions at once,
allows per-question answering after a thinking timer, delivers a report, then
enables a multi-turn voice chat for post-report review.
"""

import os
import re
import time
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
from reference_answer import generate_reference_answer

_SENTENCE_END = re.compile(r"[。？！?!\n]")


class _State(Enum):
    IDLE = "idle"
    SHOWING_QUESTIONS = "showing_questions"  # all 3 questions displayed, thinking timer active
    ANSWERING = "answering"  # user answering one specific question (mic hot)
    GENERATING_FEEDBACK = "generating_feedback"
    FINISHED = "finished"  # report delivered; idle for chat
    CHAT_LISTENING = "chat_listening"  # user speaking a chat question
    CHAT_THINKING = "chat_thinking"  # LLM + TTS in progress


THINKING_SECS = 600  # 10-minute thinking window


class InterviewController(FrameProcessor):
    """Event-driven interview controller.

    State transitions (practice):
        idle -> showing_questions   (start_practice)
        showing_questions -> answering  (start_question)
        answering -> showing_questions  (done_answering)
        showing_questions -> generating_feedback  (all answered)
        generating_feedback -> finished

    State transitions (post-report chat):
        finished -> chat_listening  (chat_start)
        chat_listening -> chat_thinking  (chat_stop, non-empty transcript)
        chat_listening -> finished  (chat_stop, empty transcript)
        chat_thinking -> finished   (LLM + TTS done)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = _State.IDLE
        self._session: PracticeSession | None = None
        self._current_transcript_parts: list[str] = []
        self._chat_transcript_parts: list[str] = []
        self._active_idx: int | None = None
        self._question_start_ts: dict[int, float] = {}
        self._ref_answers: dict[int, str] = {}
        self._ref_tasks: dict[int, object] = {}
        self._q_comments: dict[int, str] = {}
        self._q_comment_tasks: dict[int, object] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

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
        if message.get("label") != "rtvi-ai" or message.get("type") != "client-message":
            return False

        data = message.get("data", {})
        action = data.get("t", "")

        if action == "start_practice":
            await self._on_start_practice()
            return True
        if action == "skip_thinking":
            await self._on_skip_thinking()
            return True
        if action == "start_question":
            index = int((data.get("d") or {}).get("index", -1))
            await self._on_start_question(index)
            return True
        if action == "done_answering":
            await self._on_done_answering()
            return True
        if action == "chat_start":
            await self._on_chat_start()
            return True
        if action == "chat_stop":
            await self._on_chat_stop()
            return True
        return False

    # ------------------------------------------------------------------
    # Practice state handlers
    # ------------------------------------------------------------------

    async def _on_start_practice(self):
        if self._state not in (_State.IDLE, _State.FINISHED):
            return
        logger.info("Starting practice session")
        self._current_transcript_parts.clear()
        self._chat_transcript_parts.clear()
        self._active_idx = None
        self._question_start_ts.clear()
        self._ref_answers.clear()
        self._ref_tasks.clear()
        self._q_comments.clear()
        self._q_comment_tasks.clear()
        self._session = build_practice_session()
        self._state = _State.SHOWING_QUESTIONS

        for i, q in enumerate(self._session.questions):
            self._ref_tasks[i] = self.create_task(self._fetch_ref_answer(q, i), f"ref_answer_{i}")

        await self._send_ui(
            {
                "type": "questions_batch",
                "questions": [
                    {"index": i + 1, "category": q.category, "text": q.prompt}
                    for i, q in enumerate(self._session.questions)
                ],
                "thinking_secs": THINKING_SECS,
            }
        )

    async def _on_skip_thinking(self):
        if self._state != _State.SHOWING_QUESTIONS:
            return
        logger.info("User skipped thinking timer")
        await self._send_ui({"type": "thinking_ended"})

    async def _on_start_question(self, index: int):
        if self._state != _State.SHOWING_QUESTIONS or self._session is None:
            return
        idx = index - 1
        if idx < 0 or idx >= self._session.total_questions:
            return
        if self._session.is_answered(idx):
            return
        logger.info(f"User started answering Q{index}")
        self._active_idx = idx
        self._question_start_ts[idx] = time.monotonic()
        self._current_transcript_parts.clear()
        self._state = _State.ANSWERING
        await self._send_ui({"type": "waiting_answer", "index": index})

    async def _on_done_answering(self):
        if self._state != _State.ANSWERING or self._session is None or self._active_idx is None:
            return

        idx = self._active_idx
        elapsed = int(time.monotonic() - self._question_start_ts.get(idx, time.monotonic()))
        answer = " ".join(self._current_transcript_parts).strip() or "（未检测到回答）"
        self._session.record_answer(idx, answer, elapsed)
        self._current_transcript_parts.clear()
        logger.info(f"Answer recorded for Q{idx + 1}: {answer[:80]}...")

        await self._send_ui({"type": "answer_transcript", "index": idx + 1, "text": answer})

        self._active_idx = None
        self._state = _State.SHOWING_QUESTIONS

        if self._session.all_answered():
            self._state = _State.GENERATING_FEEDBACK
            await self._send_ui({"type": "generating_feedback"})
            await self._generate_and_deliver_feedback()

    # ------------------------------------------------------------------
    # Chat state handlers
    # ------------------------------------------------------------------

    async def _on_chat_start(self):
        if self._state != _State.FINISHED:
            return
        logger.info("Chat started")
        self._chat_transcript_parts.clear()
        self._state = _State.CHAT_LISTENING

    async def _on_chat_stop(self):
        if self._state != _State.CHAT_LISTENING or self._session is None:
            return
        question = " ".join(self._chat_transcript_parts).strip()
        self._chat_transcript_parts.clear()
        if not question:
            logger.info("Empty chat question, returning to finished")
            self._state = _State.FINISHED
            await self._send_ui({"type": "chat_bot_done"})
            return
        logger.info(f"Chat question: {question[:80]}")
        await self._send_ui({"type": "chat_user_transcript", "text": question})
        self._session.chat_history.append({"role": "user", "content": question})
        self._state = _State.CHAT_THINKING
        await self._handle_chat_query()

    # ------------------------------------------------------------------
    # Transcript accumulation (routes based on state)
    # ------------------------------------------------------------------

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if not frame.text.strip():
            return
        if self._state == _State.ANSWERING:
            self._current_transcript_parts.append(frame.text.strip())
            await self._send_ui(
                {"type": "transcript", "text": " ".join(self._current_transcript_parts)}
            )
        elif self._state == _State.CHAT_LISTENING:
            self._chat_transcript_parts.append(frame.text.strip())
            await self._send_ui(
                {"type": "transcript", "text": " ".join(self._chat_transcript_parts)}
            )

    # ------------------------------------------------------------------
    # Reference answer pre-fetch
    # ------------------------------------------------------------------

    async def _fetch_ref_answer(self, q, index: int):
        try:
            ref = await generate_reference_answer(q, index)
            self._ref_answers[index] = ref.answer
            logger.info(f"Reference answer ready for Q{index + 1}")
        except Exception as e:
            logger.error(f"Reference answer generation failed for Q{index + 1}: {e}")

    # ------------------------------------------------------------------
    # Per-question comment
    # ------------------------------------------------------------------

    async def _generate_question_comment(self, q, answer: str, index: int):
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

        for i, q in enumerate(self._session.questions):
            answer = self._session.answers.get(i, "")
            self._q_comment_tasks[i] = self.create_task(
                self._generate_question_comment(q, answer, i), f"q_comment_{i}"
            )

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

        overall, recommendations = self._parse_summary(feedback_text)

        all_bg_tasks = list(self._ref_tasks.values()) + list(self._q_comment_tasks.values())
        pending = [t for t in all_bg_tasks if isinstance(t, asyncio.Task) and not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        n = len(self._session.questions)
        self._session.reference_answers = {i: self._ref_answers.get(i, "") for i in range(n)}
        self._session.question_comments = {i: self._q_comments.get(i, "") for i in range(n)}

        report_questions = []
        for i, q in enumerate(self._session.questions):
            elapsed = self._session.answer_times.get(i, 0)
            mm, ss = divmod(elapsed, 60)
            report_questions.append(
                {
                    "index": i + 1,
                    "category": q.category,
                    "prompt": q.prompt,
                    "answer": self._session.answers.get(i, ""),
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
        await self._send_ui({"type": "chat_ready"})

    # ------------------------------------------------------------------
    # Chat query — LLM with sentence-level TTS
    # ------------------------------------------------------------------

    def _build_report_context(self) -> str:
        if self._session is None:
            return ""
        lines = []
        for i, q in enumerate(self._session.questions):
            lines.append(f"第{i + 1}题【{q.category}】")
            lines.append(f"题目：{q.prompt}")
            lines.append(f"考生回答：{self._session.answers.get(i, '未作答')}")
            ref = self._ref_answers.get(i, "")
            if ref:
                lines.append(f"参考答案：{ref}")
            comment = self._q_comments.get(i, "")
            if comment:
                lines.append(f"逐题点评：{comment}")
            elapsed = self._session.answer_times.get(i, 0)
            mm, ss = divmod(elapsed, 60)
            lines.append(f"答题用时：{mm}分{ss:02d}秒")
            lines.append("")
        if self._session.final_feedback:
            overall, recs = self._parse_summary(self._session.final_feedback)
            if overall:
                lines.append(f"总体评价：{overall}")
            if recs:
                lines.append(f"改进建议：{recs}")
        return "\n".join(lines)

    async def _handle_chat_query(self):
        context = self._build_report_context()
        system_content = (
            "你是一名广州事业单位面试陪练教练。考生刚完成本次三道题练习，"
            "请基于以下题目、答题、参考答案和点评，与考生进行有针对性的复盘对话。"
            "回答要简洁、口语化，不超过200字，不要使用markdown、emoji或项目符号。"
            f"\n\n本次练习内容：\n{context}"
        )
        messages = [{"role": "system", "content": system_content}, *self._session.chat_history]

        response_parts: list[str] = []
        sentence_buf = ""
        try:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("ARK_API_KEY"),
                base_url=os.getenv("ARK_BASE_URL"),
            )
            stream = await client.chat.completions.create(
                model=os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"),
                messages=messages,
                extra_body={"thinking": {"type": "disabled"}},
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                response_parts.append(delta)
                await self._send_ui({"type": "chat_bot_chunk", "text": delta})
                sentence_buf += delta
                # Flush complete sentences to TTS immediately
                while True:
                    m = _SENTENCE_END.search(sentence_buf)
                    if not m:
                        break
                    sentence = sentence_buf[: m.end()].strip()
                    sentence_buf = sentence_buf[m.end() :]
                    if sentence:
                        await self.push_frame(TTSSpeakFrame(text=sentence))
            # Flush any remaining text
            if sentence_buf.strip():
                await self.push_frame(TTSSpeakFrame(text=sentence_buf.strip()))
        except Exception as e:
            logger.error(f"Chat LLM call failed: {e}")
            err_msg = "抱歉，对话生成失败，请重试。"
            response_parts.append(err_msg)
            await self._send_ui({"type": "chat_bot_chunk", "text": err_msg})

        full_response = "".join(response_parts)
        self._session.chat_history.append({"role": "assistant", "content": full_response})
        await self._send_ui({"type": "chat_bot_done"})
        self._state = _State.FINISHED
        logger.info("Chat response complete")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_summary(text: str) -> tuple[str, str]:
        parts = re.split(r"\[([^\]]+)\]", text)
        sections: dict[str, str] = {}
        for i in range(1, len(parts) - 1, 2):
            sections[parts[i].strip()] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        overall = sections.get("总体评价", text.strip())
        recommendations = sections.get("改进建议", "")
        return overall, recommendations

    async def _send_ui(self, data: dict):
        msg = {"label": "rtvi-ai", "type": "server-message", "data": data}
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=msg),
            FrameDirection.DOWNSTREAM,
        )
