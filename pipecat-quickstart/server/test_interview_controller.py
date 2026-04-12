"""Tests for InterviewController state machine and pipeline behaviour."""

import unittest

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from interview_controller import InterviewController, _State
from interview_practice import _STATE_PATH


def _make_rtvi_client_message(action: str, data=None):
    """Build an InputTransportMessageFrame wrapping an RTVI client-message."""
    return InputTransportMessageFrame(
        message={
            "label": "rtvi-ai",
            "type": "client-message",
            "id": "test-id",
            "data": {"t": action, "d": data},
        }
    )


async def _start_and_transition_to_waiting(c: InterviewController):
    """Send start_practice, then simulate TTSStoppedFrame to enter WAITING_ANSWER."""
    await run_test(c, frames_to_send=[_make_rtvi_client_message("start_practice")])
    assert c._state == _State.READING_QUESTION
    assert c._waiting_for_tts_done is True
    c._state = _State.WAITING_ANSWER
    c._waiting_for_tts_done = False
    c._current_transcript_parts.clear()


class TestControllerStateMachine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _STATE_PATH.unlink(missing_ok=True)

    def tearDown(self):
        _STATE_PATH.unlink(missing_ok=True)

    async def test_initial_state_is_idle(self):
        c = InterviewController()
        self.assertEqual(c._state, _State.IDLE)

    async def test_start_practice_transitions_to_reading(self):
        """After start_practice, controller should be in READING_QUESTION waiting for TTS."""
        c = InterviewController()
        start_msg = _make_rtvi_client_message("start_practice")

        (down, _) = await run_test(c, frames_to_send=[start_msg])

        self.assertEqual(c._state, _State.READING_QUESTION)
        self.assertTrue(c._waiting_for_tts_done)
        tts_frames = [f for f in down if isinstance(f, TTSSpeakFrame)]
        self.assertGreater(len(tts_frames), 0, "Should push TTSSpeakFrame for question")

        ui_frames = [f for f in down if isinstance(f, OutputTransportMessageUrgentFrame)]
        self.assertGreater(len(ui_frames), 0, "Should push UI updates")

    async def test_bot_stopped_speaking_transitions_to_waiting_answer(self):
        """BotStoppedSpeakingFrame upstream should transition to WAITING_ANSWER."""
        c = InterviewController()
        start_msg = _make_rtvi_client_message("start_practice")
        await run_test(c, frames_to_send=[start_msg])
        self.assertEqual(c._state, _State.READING_QUESTION)
        self.assertTrue(c._waiting_for_tts_done)

        from pipecat.processors.frame_processor import FrameDirection

        bot_stopped = BotStoppedSpeakingFrame()
        (down, _) = await run_test(
            c,
            frames_to_send=[bot_stopped],
            frames_to_send_direction=FrameDirection.UPSTREAM,
        )

        self.assertEqual(c._state, _State.WAITING_ANSWER)
        self.assertFalse(c._waiting_for_tts_done)

    async def test_start_practice_ignored_when_not_idle(self):
        """Double start should not crash or restart."""
        c = InterviewController()
        start1 = _make_rtvi_client_message("start_practice")
        start2 = _make_rtvi_client_message("start_practice")

        (down, _) = await run_test(
            c,
            frames_to_send=[start1, SleepFrame(sleep=0.05), start2],
        )
        tts_frames = [f for f in down if isinstance(f, TTSSpeakFrame)]
        self.assertEqual(len(tts_frames), 1, "Second start should be ignored")

    async def test_transcript_accumulation(self):
        """In WAITING_ANSWER state, transcription frames should be accumulated."""
        c = InterviewController()
        await _start_and_transition_to_waiting(c)

        t1 = TranscriptionFrame(text="我认为", user_id="u1", timestamp="t1")
        t2 = TranscriptionFrame(text="这个问题很重要", user_id="u1", timestamp="t2")

        await run_test(c, frames_to_send=[t1, t2])
        self.assertEqual(len(c._current_transcript_parts), 2)
        self.assertIn("我认为", c._current_transcript_parts[0])

    async def test_done_answering_records_and_moves_to_next(self):
        """done_answering should record answer and read next question."""
        c = InterviewController()
        await _start_and_transition_to_waiting(c)

        t1 = TranscriptionFrame(text="回答内容", user_id="u1", timestamp="t1")
        done = _make_rtvi_client_message("done_answering")

        (down, _) = await run_test(
            c,
            frames_to_send=[t1, SleepFrame(sleep=0.05), done],
        )

        self.assertIsNotNone(c._session)
        self.assertEqual(len(c._session.answers), 1)
        self.assertEqual(c._session.answers[0], "回答内容")

        tts_frames = [f for f in down if isinstance(f, TTSSpeakFrame)]
        self.assertEqual(len(tts_frames), 1, "Should push TTSSpeakFrame for next question")

    async def test_done_answering_ignored_when_not_waiting(self):
        """done_answering in idle state should be a no-op."""
        c = InterviewController()
        done = _make_rtvi_client_message("done_answering")

        (down, _) = await run_test(c, frames_to_send=[done])
        tts_frames = [f for f in down if isinstance(f, TTSSpeakFrame)]
        self.assertEqual(len(tts_frames), 0)

    async def test_non_rtvi_messages_pass_through(self):
        """Non-RTVI transport messages should be forwarded unchanged."""
        c = InterviewController()
        other = InputTransportMessageFrame(message={"type": "other", "data": "hello"})

        (down, _) = await run_test(c, frames_to_send=[other])
        forwarded = [f for f in down if isinstance(f, InputTransportMessageFrame)]
        self.assertEqual(len(forwarded), 1)

    async def test_transcription_passes_through(self):
        """TranscriptionFrame should be forwarded downstream (for other processors)."""
        c = InterviewController()
        await _start_and_transition_to_waiting(c)

        t = TranscriptionFrame(text="test", user_id="u1", timestamp="t")
        (down, _) = await run_test(c, frames_to_send=[t])
        transcriptions = [f for f in down if isinstance(f, TranscriptionFrame)]
        self.assertEqual(len(transcriptions), 1)


class TestControllerUIMessages(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _STATE_PATH.unlink(missing_ok=True)

    def tearDown(self):
        _STATE_PATH.unlink(missing_ok=True)

    async def test_start_sends_progress_and_question(self):
        """On start, controller should send progress + question UI messages."""
        c = InterviewController()
        start = _make_rtvi_client_message("start_practice")

        (down, _) = await run_test(c, frames_to_send=[start])

        ui_messages = []
        for f in down:
            if isinstance(f, OutputTransportMessageUrgentFrame):
                msg = f.message
                if isinstance(msg, dict) and msg.get("type") == "server-message":
                    ui_messages.append(msg.get("data", {}))

        types = [m.get("type") for m in ui_messages]
        self.assertIn("progress", types)
        self.assertIn("question", types)
        self.assertNotIn("waiting_answer", types, "waiting_answer deferred until TTS completes")


if __name__ == "__main__":
    unittest.main()
