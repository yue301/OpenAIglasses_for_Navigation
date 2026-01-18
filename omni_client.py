# omni_client.py
# -*- coding: utf-8 -*-
import os, base64, json
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
import aiohttp  # 新增异步HTTP依赖

# ===== DashScope 配置（替换原OpenAI SDK）=====
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a9440db694924559ae4ebdc2023d2b9a")
if not API_KEY:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量")

QWEN_MODEL = "qwen-omni-turbo"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class OmniStreamPiece:
    """对外的统一增量数据：text/audio 二选一或同时。"""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    发起一轮 Omni-Turbo ChatCompletions 流式对话：
    - content_list: OpenAI chat 的 content，多模态（image_url/text）
    - 以 stream=True 返回
    - 增量产出：OmniStreamPiece(text_delta=?, audio_b64=?)
    """
    # 构造 DashScope 兼容模式请求体（包含扩展参数）
    request_body = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": content_list}],
        "modalities": ["text", "audio"],  # DashScope 扩展参数
        "audio": {"voice": voice, "format": audio_format},  # DashScope 扩展参数
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    # 请求头（必填）
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 发起异步流式请求
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url=f"{DASHSCOPE_BASE_URL}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=60)  # 超时保护
            ) as response:
                if response.status != 200:
                    err_text = await response.text()
                    raise RuntimeError(f"DashScope请求失败 {response.status}: {err_text}")

                # 逐行解析流式响应
                async for line in response.content:
                    if not line:
                        continue
                    line = line.strip()
                    if line.startswith(b"data: "):
                        line = line[6:]  # 去掉 "data: " 前缀
                        if line == b"[DONE]":
                            break
                        
                        # 解析JSON分片
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            continue  # 忽略解析失败的分片

                        # 提取文本和音频增量
                        text_delta = None
                        audio_b64 = None
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            c0 = chunk["choices"][0]
                            delta = c0.get("delta", {})
                            
                            # 文本增量
                            if "content" in delta and delta["content"]:
                                text_delta = delta["content"]
                            
                            # 音频分片（base64）
                            if "audio" in delta and delta["audio"].get("data"):
                                audio_b64 = delta["audio"]["data"]
                            
                            # 兜底：从message中提取音频
                            if not audio_b64 and "message" in c0 and "audio" in c0["message"]:
                                audio_b64 = c0["message"]["audio"].get("data")

                        # 产出数据（文本/音频有一个就返回）
                        if text_delta or audio_b64:
                            yield OmniStreamPiece(text_delta=text_delta, audio_b64=audio_b64)
        except Exception as e:
            print(f"[OMNI CLIENT ERROR] {str(e)}")
            raise  # 抛出异常让上层处理