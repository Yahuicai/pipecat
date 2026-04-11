#!/usr/bin/env python3
"""MiniMax WebSocket TTS 官方示例测试

基于 https://platform.minimaxi.com/docs/guides/speech-t2a-websocket
将音频保存到 test_output_ws.wav

用法:
    uv run test_tts_ws.py
"""

import asyncio
import json
import os
import ssl
import wave

import websockets
from dotenv import load_dotenv

load_dotenv(override=True)

WS_URL = "wss://api.minimaxi.com/ws/v1/t2a_v2"
TEST_TEXT = "你好，我是一个语音对话助手，很高兴见到你！"
SAMPLE_RATE = 16000


def write_wav(filename: str, pcm_data: bytes, sample_rate: int):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    print(
        f"[OK] 保存音频: {filename} ({len(pcm_data)} bytes, 约 {len(pcm_data) / (sample_rate * 2):.2f}s)"
    )


async def test_websocket_tts():
    api_key = os.getenv("MINIMAX_API_KEY", "")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")

    if not api_key:
        print("[ERROR] 缺少 MINIMAX_API_KEY")
        return
    if not group_id:
        print("[ERROR] 缺少 MINIMAX_GROUP_ID")
        return

    print(f"[INFO] Group ID: {group_id}")
    print(f"[INFO] API Key 前20字符: {api_key[:20]}...")
    print(f"[INFO] WebSocket URL: {WS_URL}")
    print(f"[INFO] 测试文本: {TEST_TEXT}")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    ssl_ctx = ssl.create_default_context()

    all_audio = bytearray()
    chunk_count = 0

    try:
        print("\n[INFO] 连接 WebSocket...")
        async with websockets.connect(WS_URL, additional_headers=headers, ssl=ssl_ctx) as ws:
            print("[OK] WebSocket 已连接")

            # 1. 发送 task_start
            task_start = {
                "event": "task_start",
                "model": "speech-02-turbo",
                "voice_setting": {
                    "voice_id": "female-shaonv",
                    "speed": 1.0,
                    "vol": 1.0,
                    "pitch": 0,
                },
                "audio_setting": {
                    "format": "pcm",
                    "sample_rate": SAMPLE_RATE,
                    "bitrate": 128000,
                    "channel": 1,
                },
            }
            await ws.send(json.dumps(task_start))
            print("[INFO] 发送 task_start")

            # 2. 发送文本
            task_continue = {
                "event": "task_continue",
                "text": TEST_TEXT,
            }
            await ws.send(json.dumps(task_continue))
            print("[INFO] 发送文本")

            # 3. 发送 task_finish
            task_finish = {"event": "task_finish"}
            await ws.send(json.dumps(task_finish))
            print("[INFO] 发送 task_finish，等待音频...\n")

            # 4. 接收响应
            async for message in ws:
                try:
                    data = json.loads(message)
                    event = data.get("event", "")

                    if event == "task_started":
                        print(f"[INFO] 任务已开始: trace_id={data.get('trace_id', '')}")

                    elif event == "task_failed":
                        print(f"[ERROR] 任务失败: {data}")
                        break

                    elif "data" in data:
                        audio_hex = data["data"].get("audio", "")
                        is_final = data["data"].get("is_final", False)

                        if audio_hex:
                            audio_bytes = bytes.fromhex(audio_hex)
                            all_audio.extend(audio_bytes)
                            chunk_count += 1
                            print(
                                f"[INFO] 音频块 #{chunk_count}: {len(audio_bytes)} bytes "
                                f"(累计: {len(all_audio)} bytes, is_final={is_final})"
                            )

                        if is_final:
                            print("[INFO] 收到最终块，结束")
                            break

                    else:
                        print(f"[INFO] 其他消息: {json.dumps(data)[:200]}")

                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON 解析失败: {e}, raw: {str(message)[:100]}")

    except Exception as e:
        print(f"[ERROR] WebSocket 错误: {type(e).__name__}: {e}")
        return

    if all_audio:
        print(f"\n[OK] 共收到 {len(all_audio)} bytes 音频 ({chunk_count} 块)")
        write_wav("test_output_ws.wav", bytes(all_audio), SAMPLE_RATE)
        print("[OK] MiniMax WebSocket TTS 工作正常！")
    else:
        print("\n[ERROR] 未收到任何音频数据")


if __name__ == "__main__":
    asyncio.run(test_websocket_tts())
