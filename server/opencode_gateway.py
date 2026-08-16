import json
import time
import uuid

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


# ============================================================
# 配置
# ============================================================

MODEL_PATH = "/home/tjwei/llms/Qwen3-0.6B/"

MODEL_ID = "qwen3-0.6b"

# 你现有 nano-vLLM FastAPI
BACKEND_URL = "http://127.0.0.1:8000/generate"


# ============================================================
# tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="nano-vLLM OpenAI Gateway",
    version="0.1.0",
)


# ============================================================
# OpenAI Chat 格式
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID

    messages: list[ChatMessage]

    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int | None = None

    max_completion_tokens: int | None = None

    stream: bool = False

    # OpenCode 后面会传 tools。
    # 第一阶段先接收，但暂时不处理。
    tools: list[dict] | None = None

    tool_choice: object | None = None


# ============================================================
# model list
# ============================================================

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# ============================================================
# messages -> Qwen3 prompt
# ============================================================

def build_prompt(
    messages: list[ChatMessage],
) -> str:

    chat_messages = []

    for message in messages:

        if message.content is None:
            content = ""
        else:
            content = message.content

        chat_messages.append(
            {
                "role": message.role,
                "content": content,
            }
        )

    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,

        # 第一阶段先关闭 thinking，
        # 对 0.6B 模型以及 2048 context 更友好。
        enable_thinking=False,
    )

    return prompt


# ============================================================
# 调用你原来的 nano-vLLM /generate
# ============================================================

async def call_backend(
    request: ChatCompletionRequest,
):

    prompt = build_prompt(
        request.messages
    )

    max_tokens = (
        request.max_completion_tokens
        or request.max_tokens
        or 256
    )

    payload = {
        "prompt": prompt,
        "temperature": max(
            request.temperature,
            0.01,
        ),
        "max_tokens": min(
            max_tokens,
            512,
        ),
        "ignore_eos": False,
    }

    async with httpx.AsyncClient(
        timeout=None
    ) as client:

        response = await client.post(
            BACKEND_URL,
            json=payload,
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=502,
            detail=(
                "nano-vLLM backend failed: "
                + response.text
            ),
        )

    return response.json()


# ============================================================
# /v1/chat/completions
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
):

    request_id = (
        "chatcmpl-"
        + uuid.uuid4().hex
    )

    created = int(time.time())

    result = await call_backend(
        request
    )

    text = result["text"]

    output_tokens = result.get(
        "output_tokens",
        0,
    )

    # --------------------------------------------------------
    # 普通 JSON 返回
    # --------------------------------------------------------

    if not request.stream:

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,

            "choices": [
                {
                    "index": 0,

                    "message": {
                        "role": "assistant",
                        "content": text,
                    },

                    "finish_reason": "stop",
                }
            ],

            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": output_tokens,
                "total_tokens": output_tokens,
            },
        }

    # --------------------------------------------------------
    # OpenAI SSE
    #
    # 注意：
    # 目前属于“兼容式 streaming”：
    # nano-vLLM 完整生成后一次发给 OpenCode。
    #
    # 后面我们再把它升级成 token streaming。
    # --------------------------------------------------------

    async def generate_stream():

        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,

            "choices": [
                {
                    "index": 0,

                    "delta": {
                        "role": "assistant",
                        "content": text,
                    },

                    "finish_reason": None,
                }
            ],
        }

        yield (
            "data: "
            + json.dumps(
                first_chunk,
                ensure_ascii=False,
            )
            + "\n\n"
        )

        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,

            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        yield (
            "data: "
            + json.dumps(
                final_chunk,
                ensure_ascii=False,
            )
            + "\n\n"
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
    )
