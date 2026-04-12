"""Custom FastAPI entry point for the interview practice assistant.

Serves:
  - /            -> redirect to /client/
  - /client/     -> custom interview UI (static files)
  - /api/offer   -> WebRTC signaling (POST = SDP offer, PATCH = ICE candidates)

Usage:
    uv run main.py
    # then open http://localhost:7860/client/ in your browser
"""

import argparse
import asyncio
from pathlib import Path

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
