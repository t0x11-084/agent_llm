import json
import os
from pathlib import Path

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
)


# ============================================================
# 1. 第三方中转站配置
# ============================================================

THIRD_PARTY_BASE_URL = "https://mdkj.lol/v1"

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "low"


# ============================================================
# 2. 获取 API Key
# ============================================================

def load_api_key() -> str:

    # 优先环境变量
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key

    # 没有环境变量时读取 Codex
    auth_file = Path.home() / ".codex" / "auth.json"

    if not auth_file.exists():
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set "
            "and ~/.codex/auth.json does not exist"
        )

    with auth_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        auth = json.load(f)

    api_key = auth.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in ~/.codex/auth.json"
        )

    return api_key


# ============================================================
# 3. 创建 OpenAI Client
# ============================================================

client = AsyncOpenAI(

    api_key=load_api_key(),

    # 注意这里使用 /v1
    base_url=THIRD_PARTY_BASE_URL,

    timeout=180.0,

    # 关键：
    # 某些 Sub2API / Cloudflare WAF
    # 会拦截 OpenAI/Python User-Agent
    default_headers={
        "User-Agent": "curl/8.5.0",
        "Accept": "application/json",
    },
)


# ============================================================
# 4. 第三方 GPT 推理
# ============================================================

async def generate_with_gpt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:

    try:

        response = await client.responses.create(

            model=model,

            input=prompt,

            reasoning={
                "effort": reasoning_effort,
            },

            store=False,
        )

    except APIStatusError as exc:

        # 打印第三方真正返回的信息
        print("=" * 60)
        print("Third-party API HTTP ERROR")
        print("status_code:", exc.status_code)
        print("response:", exc.response.text)
        print("=" * 60)

        raise RuntimeError(
            f"HTTP {exc.status_code}: "
            f"{exc.response.text}"
        ) from exc

    except APIConnectionError as exc:

        print("=" * 60)
        print("Third-party API CONNECTION ERROR")
        print(str(exc))
        print("=" * 60)

        raise RuntimeError(
            f"Connection failed: {exc}"
        ) from exc

    return {
        "text": response.output_text,
        "model": response.model,
        "response_id": response.id,
    
