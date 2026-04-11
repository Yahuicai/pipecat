#!/usr/bin/env python3
"""MiniMax TTS 独立测试脚本

测试 MiniMax TTS 是否正常工作，将音频保存到 test_output.wav

用法:
    uv run test_tts.py
"""

import asyncio
import json
import os
import struct
import wave

import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_URL = "https://api.minimaxi.chat/v1/t2a_v2"
TEST_TEXT = "你好，我是一个语音对话助手，很高兴见到你！"
SAMPLE_RATE = 16000  # pipecat 默认采样率


def write_wav(filename: str, pcm_data: bytes, sample_rate: int):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    print(f"[OK] 已保存音频文件: {filename} ({len(pcm_data)} bytes PCM)")


async def test_minimax_tts():
    api_key = os.getenv("MINIMAX_API_KEY", "")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")

    if not api_key:
        print("[ERROR] 缺少 MINIMAX_API_KEY 环境变量")
        return
    if not group_id:
        print("[ERROR] 缺少 MINIMAX_GROUP_ID 环境变量")
        return

    url = f"{BASE_URL}?GroupId={group_id}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "stream": True,
        "voice_setting": {
            "voice_id": "female-shaonv",
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "bitrate": 128000,
            "format": "pcm",
            "channel": 1,
            "sample_rate": SAMPLE_RATE,
        },
        "model": "speech-02-turbo",
        "text": TEST_TEXT,
    }

    print(f"[INFO] 请求 URL: {url}")
    print(f"[INFO] 测试文本: {TEST_TEXT}")
    print(f"[INFO] 采样率: {SAMPLE_RATE} Hz")

    all_audio = bytearray()
    chunk_count = 0
    buffer = bytearray()

    async with aiohttp.ClientSession() as session:
        print("[INFO] 发送 TTS 请求...")
        async with session.post(url, headers=headers, json=payload) as response:
            print(f"[INFO] HTTP 响应状态: {response.status}")
            print(f"[INFO] Content-Type: {response.headers.get('Content-Type', 'unknown')}")

            if response.status != 200:
                body = await response.text()
                print(f"[ERROR] HTTP 错误: {body}")
                return

            raw_chunks = []
            async for chunk in response.content.iter_chunked(65536):
                if not chunk:
                    continue
                raw_chunks.append(chunk)
                buffer.extend(chunk)

            # 打印原始响应（前 2000 字节）
            full_raw = b"".join(raw_chunks)
            print(f"\n[DEBUG] 原始响应总长度: {len(full_raw)} bytes")
            print(f"[DEBUG] 原始响应前 1000 字节 (repr):")
            print(repr(full_raw[:1000]))
            print()

            # 重新解析
            buffer = bytearray(full_raw)
            while b"data:" in buffer:
                start = buffer.find(b"data:")
                next_start = buffer.find(b"data:", start + 5)

                if next_start == -1:
                    # 最后一个 data 块，尝试直接解析（不需要下一个 data: 标记）
                    data_block = buffer[start:]
                    buffer = bytearray()
                    try:
                        text = data_block[5:].decode("utf-8").strip()
                        if text:
                            data = json.loads(text)
                            if "extra_info" in data:
                                ei = data["extra_info"]
                                print(
                                    f"[INFO] extra_info (最后块): audio_length={ei.get('audio_length')}ms"
                                )
                            else:
                                chunk_data = data.get("data", {})
                                audio_hex = chunk_data.get("audio") if chunk_data else None
                                if audio_hex:
                                    audio_bytes = bytes.fromhex(audio_hex)
                                    all_audio.extend(audio_bytes)
                                    chunk_count += 1
                                    print(
                                        f"[INFO] 收到音频块 #{chunk_count} (最后块): {len(audio_bytes)} bytes"
                                    )
                    except Exception as e:
                        print(f"[WARN] 最后块解析失败: {e}")
                    break

                data_block = buffer[start:next_start]
                buffer = buffer[next_start:]

                try:
                    data = json.loads(data_block[5:].decode("utf-8"))

                    if "extra_info" in data:
                        ei = data["extra_info"]
                        print(
                            f"[INFO] extra_info: audio_length={ei.get('audio_length')}ms, "
                            f"word_count={ei.get('word_count')}"
                        )
                        continue

                    chunk_data = data.get("data", {})
                    if not chunk_data:
                        print(f"[WARN] data 块为空: {data}")
                        continue

                    audio_hex = chunk_data.get("audio")
                    if not audio_hex:
                        print(f"[WARN] 无 audio 字段: {chunk_data}")
                        continue

                    audio_bytes = bytes.fromhex(audio_hex)
                    all_audio.extend(audio_bytes)
                    chunk_count += 1
                    print(
                        f"[INFO] 收到音频块 #{chunk_count}: {len(audio_bytes)} bytes "
                        f"(累计: {len(all_audio)} bytes)"
                    )

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[WARN] 解析错误: {e}, block: {data_block[:120]}")
                    continue

    if all_audio:
        print(f"\n[OK] 总计收到 {len(all_audio)} bytes 音频数据 ({chunk_count} 个块)")
        print(f"[INFO] 音频时长约: {len(all_audio) / (SAMPLE_RATE * 2):.2f} 秒")
        output_file = "test_output.wav"
        write_wav(output_file, bytes(all_audio), SAMPLE_RATE)
        print(f"\n[OK] MiniMax TTS 工作正常！请用音频播放器打开 {output_file} 验证音质。")
    else:
        print("\n[ERROR] 未收到任何音频数据！")
        print("可能原因:")
        print("  1. API Key 或 Group ID 不正确")
        print("  2. 网络无法访问 api.minimaxi.chat")
        print("  3. 账户余额不足")


if __name__ == "__main__":
    asyncio.run(test_minimax_tts())
