"""Tests for interview_practice module.

Covers: question bank loading, used-ID state, session lifecycle,
InterviewPracticeProcessor pipeline behaviour, and record persistence.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interview_practice import (
    _STATE_PATH,
    InterviewPracticeProcessor,
    PracticeQuestion,
    PracticeSession,
    _load_question_bank,
    build_final_feedback_prompt,
    build_follow_up_prompt,
    build_practice_session,
    build_start_prompt,
    save_practice_record,
)

# ---------------------------------------------------------------------------
# Question bank & state persistence
# ---------------------------------------------------------------------------


class TestQuestionBank(unittest.TestCase):
    def test_load_question_bank_structure(self):
        bank = _load_question_bank()
        self.assertEqual(len(bank), 3)
        for category, items in bank.items():
            self.assertGreater(len(items), 0)
            for q in items:
                self.assertIn("id", q)
                self.assertIn("prompt", q)

    def test_build_session_selects_one_per_category(self):
        session = build_practice_session()
        self.assertEqual(session.total_questions, 3)
        categories = {q.category for q in session.questions}
        self.assertEqual(len(categories), 3)

    def test_used_ids_avoid_repeat(self):
        """Two consecutive sessions should not share any question ID
        (unless a category is exhausted, which needs 5 rounds)."""
        _STATE_PATH.unlink(missing_ok=True)
        try:
            s1 = build_practice_session()
            s2 = build_practice_session()
            ids1 = {q.id for q in s1.questions}
            ids2 = {q.id for q in s2.questions}
            self.assertTrue(ids1.isdisjoint(ids2), f"Overlap: {ids1 & ids2}")
        finally:
            _STATE_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PracticeSession unit tests
# ---------------------------------------------------------------------------


def _make_session() -> PracticeSession:
    return PracticeSession(
        questions=[
            PracticeQuestion(id="a1", category="综合分析", prompt="题目A"),
            PracticeQuestion(id="b1", category="计划组织/人际沟通", prompt="题目B"),
            PracticeQuestion(id="c1", category="应急应变/情景模拟", prompt="题目C"),
        ]
    )


class TestPracticeSession(unittest.TestCase):
    def test_initial_state(self):
        s = _make_session()
        self.assertEqual(s.total_questions, 3)
        self.assertEqual(s.current_question_index, 0)
        self.assertFalse(s.is_complete)
        self.assertIsNone(s.final_feedback)

    def test_record_answers(self):
        s = _make_session()
        s.record_answer("回答1")
        self.assertEqual(s.current_question_index, 1)
        s.record_answer("回答2")
        s.record_answer("回答3")
        self.assertTrue(s.is_complete)

    def test_record_answer_ignores_empty(self):
        s = _make_session()
        s.record_answer("")
        s.record_answer("   ")
        self.assertEqual(s.current_question_index, 0)

    def test_record_answer_stops_at_limit(self):
        s = _make_session()
        for i in range(5):
            s.record_answer(f"answer {i}")
        self.assertEqual(len(s.answers), 3)

    def test_formatted_history(self):
        s = _make_session()
        s.record_answer("回答1")
        history = s.formatted_history()
        self.assertIn("题目A", history)
        self.assertIn("回答1", history)
        self.assertIn("用户尚未作答", history)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestPromptBuilders(unittest.TestCase):
    def test_start_prompt_contains_all_questions(self):
        s = _make_session()
        prompt = build_start_prompt(s)
        for q in s.questions:
            self.assertIn(q.prompt, prompt)
            self.assertIn(q.category, prompt)

    def test_follow_up_prompt(self):
        s = _make_session()
        s.record_answer("回答1")
        prompt = build_follow_up_prompt(s)
        self.assertIn("题目B", prompt)

    def test_final_feedback_prompt(self):
        s = _make_session()
        for i in range(3):
            s.record_answer(f"回答{i + 1}")
        prompt = build_final_feedback_prompt(s)
        self.assertIn("回答1", prompt)
        self.assertIn("回答3", prompt)
        self.assertIn("总点评", prompt)


# ---------------------------------------------------------------------------
# Record persistence
# ---------------------------------------------------------------------------


class TestSavePracticeRecord(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_records_dir = Path(__file__).resolve().parent / "practice_records"

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_creates_md_file(self):
        s = _make_session()
        s.record_answer("回答1")
        s.record_answer("回答2")
        s.record_answer("回答3")
        s.final_feedback = "总点评内容"

        tmp_records = Path(self._tmpdir) / "practice_records"
        with patch("interview_practice._RECORDS_DIR", tmp_records):
            path = save_practice_record(s)

        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8")
        self.assertIn("面试练习记录", content)
        self.assertIn("题目A", content)
        self.assertIn("回答1", content)
        self.assertIn("总点评内容", content)


# ---------------------------------------------------------------------------
# InterviewPracticeProcessor (pipeline-level)
# ---------------------------------------------------------------------------


class TestInterviewPracticeProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_processor_passes_frames_through(self):
        """The processor should forward LLMContextFrame downstream unchanged
        (while also capturing answers and injecting control messages)."""
        from pipecat.frames.frames import LLMContextFrame, TextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.tests.utils import run_test

        s = _make_session()
        processor = InterviewPracticeProcessor(s)

        ctx = LLMContext([{"role": "developer", "content": "init"}])
        frames_to_send = [LLMContextFrame(context=ctx)]

        (down, _) = await run_test(
            processor,
            frames_to_send=frames_to_send,
        )
        self.assertTrue(
            any(isinstance(f, LLMContextFrame) for f in down),
            "LLMContextFrame should pass through",
        )

    async def test_processor_captures_user_answer(self):
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.tests.utils import run_test

        s = _make_session()
        processor = InterviewPracticeProcessor(s)

        ctx = LLMContext(
            [
                {"role": "developer", "content": "init"},
                {"role": "user", "content": "用户的第一个回答"},
            ]
        )
        frames_to_send = [LLMContextFrame(context=ctx)]

        await run_test(processor, frames_to_send=frames_to_send)
        self.assertEqual(len(s.answers), 1)
        self.assertEqual(s.answers[0], "用户的第一个回答")

    async def test_processor_injects_follow_up(self):
        """After first answer, processor should inject a follow-up developer message."""
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.tests.utils import run_test

        s = _make_session()
        processor = InterviewPracticeProcessor(s)

        ctx = LLMContext(
            [
                {"role": "developer", "content": "init"},
                {"role": "user", "content": "第一题回答"},
            ]
        )
        frames_to_send = [LLMContextFrame(context=ctx)]

        (down, _) = await run_test(processor, frames_to_send=frames_to_send)
        context_frame = next(f for f in down if isinstance(f, LLMContextFrame))
        messages = context_frame.context.get_messages()
        dev_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "developer"]
        self.assertTrue(
            any("下一题" in m["content"] for m in dev_messages),
            "Should inject follow-up prompt with next question",
        )

    async def test_processor_injects_final_feedback(self):
        """After all 3 answers, processor should inject the final feedback prompt."""
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.tests.utils import SleepFrame, run_test

        s = _make_session()
        processor = InterviewPracticeProcessor(s)

        ctx1 = LLMContext(
            [
                {"role": "developer", "content": "init"},
                {"role": "user", "content": "第一题回答"},
            ]
        )
        ctx2 = LLMContext(
            [
                {"role": "developer", "content": "init"},
                {"role": "user", "content": "第一题回答"},
                {"role": "user", "content": "第二题回答"},
            ]
        )
        ctx3 = LLMContext(
            [
                {"role": "developer", "content": "init"},
                {"role": "user", "content": "第一题回答"},
                {"role": "user", "content": "第二题回答"},
                {"role": "user", "content": "第三题回答"},
            ]
        )

        frames_to_send = [
            LLMContextFrame(context=ctx1),
            SleepFrame(sleep=0.05),
            LLMContextFrame(context=ctx2),
            SleepFrame(sleep=0.05),
            LLMContextFrame(context=ctx3),
        ]

        (down, _) = await run_test(processor, frames_to_send=frames_to_send)

        last_ctx_frame = [f for f in down if isinstance(f, LLMContextFrame)][-1]
        messages = last_ctx_frame.context.get_messages()
        dev_contents = " ".join(
            m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "developer"
        )
        self.assertIn("总点评", dev_contents)
        self.assertTrue(s.final_feedback_requested)
        self.assertTrue(s.is_complete)


if __name__ == "__main__":
    unittest.main()
