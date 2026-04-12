"""Custom FastAPI entry point for the interview practice assistant.

Serves:
  - /            -> redirect to /client/
  - /client/     -> custom interview UI (static files)
  - /api/offer   -> WebRTC signaling (POST = SDP offer, PATCH = ICE candidates)
  - /api/records -> practice history (GET)

Usage:
    uv run main.py
    # then open http://localhost:7860/client/ in your browser
"""

import argparse
import re
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

load_dotenv(override=True)

_SERVER_DIR = Path(__file__).resolve().parent
_CLIENT_DIR = _SERVER_DIR.parent / "client"
_RECORDS_DIR = _SERVER_DIR / "practice_records"

app = FastAPI(title="Interview Practice Assistant")

small_webrtc_handler = SmallWebRTCRequestHandler()


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """WebRTC SDP offer -> answer."""
    from bot_interview import bot

    async def on_connection(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body=request.request_data,
        )
        background_tasks.add_task(bot, runner_args)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """WebRTC ICE candidate trickle."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


def _parse_record_md(path: Path) -> dict[str, Any]:
    """Parse a practice record markdown file into structured data."""
    text = path.read_text(encoding="utf-8")
    record: dict[str, Any] = {"filename": path.name, "questions": []}

    title_match = re.search(r"^# 面试练习记录 (.+)$", text, re.MULTILINE)
    record["title"] = title_match.group(1).strip() if title_match else path.stem

    q_blocks = re.split(r"^## 第\d+题", text, flags=re.MULTILINE)[1:]
    for block in q_blocks:
        cat_match = re.search(r"【(.+?)】", block)
        prompt_match = re.search(r"\*\*题目：\*\*\s*(.+)", block)
        answer_match = re.search(
            r"\*\*回答：\*\*\s*(.+?)(?=\n\*\*参考答案|\n---|\n## |$)", block, re.DOTALL
        )
        ref_match = re.search(r"\*\*参考答案：\*\*\s*(.+?)(?=\n---|\n## |$)", block, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else ""
        ref_text = ref_match.group(1).strip() if ref_match else ""
        record["questions"].append(
            {
                "category": cat_match.group(1) if cat_match else "",
                "prompt": prompt_match.group(1).strip() if prompt_match else "",
                "answer": answer_text,
                "reference_answer": ref_text,
            }
        )

    feedback_match = re.search(r"## 总点评\s*\n(.+)", text, re.DOTALL)
    record["feedback"] = feedback_match.group(1).strip() if feedback_match else ""

    return record


@app.get("/api/records")
async def list_records():
    """Return all practice records, newest first."""
    if not _RECORDS_DIR.exists():
        return []
    files = sorted(_RECORDS_DIR.glob("*.md"), reverse=True)
    records = []
    for f in files:
        try:
            records.append(_parse_record_md(f))
        except Exception as e:
            logger.warning(f"Failed to parse {f.name}: {e}")
    return records


app.mount("/client", StaticFiles(directory=str(_CLIENT_DIR), html=True), name="client")


def main():
    parser = argparse.ArgumentParser(description="Interview Practice Assistant")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info(f"Open http://localhost:{args.port}/client/ in your browser")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
