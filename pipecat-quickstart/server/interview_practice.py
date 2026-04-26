import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_BASE_DIR = Path(__file__).resolve().parent
_QUESTIONS_PATH = _BASE_DIR / "questions.json"
_STATE_PATH = _BASE_DIR / "practice_state.json"
_RECORDS_DIR = _BASE_DIR / "practice_records"


# ---------------------------------------------------------------------------
# Question bank (loaded from JSON)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PracticeQuestion:
    id: str
    category: str
    prompt: str


def _load_question_bank() -> dict[str, list[dict]]:
    with open(_QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Used-question state (persisted to JSON)
# ---------------------------------------------------------------------------


def _load_used_ids() -> set[str]:
    if not _STATE_PATH.exists():
        return set()
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("used_ids", []))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_used_ids(used_ids: set[str]):
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"used_ids": sorted(used_ids)}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Practice session
# ---------------------------------------------------------------------------


@dataclass
class PracticeSession:
    questions: List[PracticeQuestion]
    answers: dict[int, str] = field(default_factory=dict)
    answer_times: dict[int, int] = field(default_factory=dict)
    reference_answers: dict[int, str] = field(default_factory=dict)
    question_comments: dict[int, str] = field(default_factory=dict)
    final_feedback_requested: bool = False
    final_feedback: str | None = None
    chat_history: list[dict] = field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    def is_answered(self, index: int) -> bool:
        return index in self.answers

    def all_answered(self) -> bool:
        return len(self.answers) >= self.total_questions

    def record_answer(self, index: int, answer: str, elapsed_secs: int = 0):
        if 0 <= index < self.total_questions:
            self.answers[index] = answer.strip() or "（未检测到回答）"
            self.answer_times[index] = elapsed_secs

    def formatted_history(self) -> str:
        lines = []
        for idx, question in enumerate(self.questions):
            answer = self.answers.get(idx, "用户尚未作答")
            lines.append(f"第{idx + 1}题【{question.category}】")
            lines.append(f"题目：{question.prompt}")
            lines.append(f"回答：{answer}")
        return "\n".join(lines)


_SESSION_QUESTION_COUNT = 3


def build_practice_session() -> PracticeSession:
    """从 JSON 题库中随机选 3 个类别各抽 1 题，自动跳过已用题目。

    当某个类别的所有题目都已用过时，自动重置该类别的已用记录。
    """
    bank = _load_question_bank()
    used_ids = _load_used_ids()

    categories = list(bank.keys())
    chosen_categories = random.sample(categories, min(_SESSION_QUESTION_COUNT, len(categories)))

    selected: list[PracticeQuestion] = []
    for category in chosen_categories:
        items = bank[category]
        available = [q for q in items if q["id"] not in used_ids]
        if not available:
            for q in items:
                used_ids.discard(q["id"])
            available = items
        chosen = random.choice(available)
        selected.append(
            PracticeQuestion(id=chosen["id"], category=category, prompt=chosen["prompt"])
        )

    new_used = used_ids | {q.id for q in selected}
    _save_used_ids(new_used)

    return PracticeSession(questions=selected)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_start_prompt(session: PracticeSession) -> str:
    question_lines = [
        f"{idx}. 【{question.category}】{question.prompt}"
        for idx, question in enumerate(session.questions, start=1)
    ]
    question_text = "\n".join(question_lines)

    return (
        "你是一名广州事业单位面试中文陪练助手。你的所有输出都会被直接朗读，"
        "所以必须使用自然、简洁、适合口语播报的中文。\n\n"
        "本轮训练是个人练习版，一共严格进行三题，按下面顺序完成，不得跳题，不得改题，不得额外加题：\n"
        f"{question_text}\n\n"
        "流程要求：\n"
        "1. 第一轮输出时，先用一句简短开场说明今天练三题，然后完整读出第1题。\n"
        "2. 用户回答完第1题后，不要点评，只用一句简短过渡，然后完整读出第2题。\n"
        "3. 用户回答完第2题后，不要点评，只用一句简短过渡，然后完整读出第3题。\n"
        "4. 用户回答完第3题后，不要再提新问题，只基于三道题和三次回答给出一次统一总点评。\n"
        "5. 总点评只包括四部分：总体表现、亮点、主要问题、改进建议。\n"
        "6. 前两题之间的过渡必须非常简短，不做分析，不做评价，不重复用户答案。\n"
        "7. 全程不要使用 emoji、项目符号或复杂格式。\n"
    )


def build_follow_up_prompt(session: PracticeSession) -> str:
    next_question = session.questions[session.current_question_index]
    return (
        f"用户已经完成第{session.current_question_index}题回答。不要点评上一题。"
        "请只用一句很短的过渡，例如'收到，我们进入下一题'。"
        f"然后完整读出下一题，不要改写题意：\n【{next_question.category}】{next_question.prompt}"
    )


def build_final_feedback_prompt(session: PracticeSession) -> str:
    return (
        "用户已经完成全部三题。不要再提新问题，也不要逐题展开长篇分析。"
        "请基于下面这三道题和用户回答，给出一次统一总点评。\n\n"
        f"{session.formatted_history()}\n\n"
        "输出要求：\n"
        "1. 只输出总点评。\n"
        "2. 总点评按四部分组织：总体表现、亮点、主要问题、改进建议。\n"
        "3. 语气像个人陪练教练，直接、具体，不要空泛。\n"
        "4. 适合口语播报，控制在大约300字以内。\n"
    )


# ---------------------------------------------------------------------------
# Practice record persistence
# ---------------------------------------------------------------------------


def save_practice_record(session: PracticeSession) -> Path:
    """将本轮练习的题目、回答和总点评保存为 Markdown 文件。"""
    _RECORDS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    filepath = _RECORDS_DIR / f"{timestamp}.md"

    title = now.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# 面试练习记录 {title}\n"]

    for idx, question in enumerate(session.questions):
        answer = session.answers.get(idx, "用户尚未作答")
        ref = session.reference_answers.get(idx, "")
        elapsed = session.answer_times.get(idx, 0)
        comment = session.question_comments.get(idx, "")
        lines.append(f"## 第{idx + 1}题【{question.category}】\n")
        lines.append(f"**题目：** {question.prompt}\n")
        lines.append(f"**回答：** {answer}\n")
        if elapsed:
            mm, ss = divmod(elapsed, 60)
            lines.append(f"**用时：** {mm}分{ss:02d}秒\n")
        if ref:
            lines.append(f"**参考答案：** {ref}\n")
        if comment:
            lines.append(f"**逐题点评：** {comment}\n")

    lines.append("---\n")
    lines.append("## 总点评\n")
    lines.append(session.final_feedback or "未生成点评")
    lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Practice statistics
# ---------------------------------------------------------------------------


def get_practice_stats() -> dict:
    """返回题库练习进度统计。"""
    bank = _load_question_bank()
    used_ids = _load_used_ids()

    category_stats = []
    total = 0
    practiced = 0
    for category, items in bank.items():
        cat_total = len(items)
        cat_practiced = sum(1 for q in items if q["id"] in used_ids)
        category_stats.append({"name": category, "total": cat_total, "practiced": cat_practiced})
        total += cat_total
        practiced += cat_practiced

    return {"total": total, "practiced": practiced, "categories": category_stats}


# ---------------------------------------------------------------------------
# Pipeline processor
# ---------------------------------------------------------------------------


class InterviewPracticeProcessor(FrameProcessor):
    def __init__(self, session: PracticeSession, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._seen_user_messages = 0

    @property
    def session(self) -> PracticeSession:
        return self._session

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._capture_new_answers(frame.context)
            self._append_control_message(frame.context)

        await self.push_frame(frame, direction)

    def _capture_new_answers(self, context: LLMContext):
        user_messages = [
            message
            for message in context.get_messages()
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        new_messages = user_messages[self._seen_user_messages :]
        for message in new_messages:
            self._session.record_answer(message["content"])
        self._seen_user_messages = len(user_messages)

    def _append_control_message(self, context: LLMContext):
        answer_count = len(self._session.answers)
        if answer_count == 0:
            return

        if answer_count < self._session.total_questions:
            context.add_message(
                {"role": "developer", "content": build_follow_up_prompt(self._session)}
            )
            return

        if not self._session.final_feedback_requested:
            context.add_message(
                {"role": "developer", "content": build_final_feedback_prompt(self._session)}
            )
            self._session.final_feedback_requested = True
