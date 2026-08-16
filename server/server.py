from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams

from third_party_api import generate_with_gpt

# ============================================================
# 1. nano-vLLM 配置
# ============================================================

MODEL_PATH = "/home/tjwei/projects/nano-vllm/llms"

MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 4


# ============================================================
# 2. 全局保存一个 nano-vLLM 实例
# ============================================================

llm: LLM | None = None


# nano-vLLM 当前的 generate() 并不是专门为多个 HTTP
# 线程同时调用设计的，所以第一版使用锁保护。
llm_lock = Lock()


# ============================================================
# 3. 定义 POST /generate 的 JSON 请求格式
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)

    temperature: float = Field(
        default=0.6,
        gt=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=128,
        ge=1,
        le=512,
    )

    ignore_eos: bool = False


# ============================================================
# 4. 定义响应格式
# ============================================================

class GenerateResponse(BaseModel):
    text: str
    output_tokens: int



# ============================================================
# 第三方 GPT 请求格式
# ============================================================

class GPTGenerateRequest(BaseModel):

    prompt: str = Field(
        min_length=1,
    )

    model: str = "gpt-5.2"

    reasoning_effort: str = "low"


class GPTGenerateResponse(BaseModel):

    text: str

    model: str

    response_id: str



# ============================================================
# 5. FastAPI 生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm

    print("Loading nano-vLLM model...")

    llm = LLM(
        MODEL_PATH,
        enforce_eager=False,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
    )

    print("nano-vLLM model loaded.")

    yield

    print("FastAPI server shutting down...")


# ============================================================
# 6. 创建 FastAPI Server
# ============================================================

app = FastAPI(
    title="nano-vLLM Server",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# 7. 健康检查
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": llm is not None,
    }


# ============================================================
# 8. LLM 推理接口
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )

    try:
        with llm_lock:
            output = llm.generate(
                [request.prompt],
                sampling_params,
                use_tqdm=False,
            )[0]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return GenerateResponse(
        text=output["text"],
        output_tokens=len(output["token_ids"]),
    )

# ============================================================
# 9. 第三方 GPT 推理接口
# ============================================================

@app.post(
    "/generate/gpt",
    response_model=GPTGenerateResponse,
)
async def generate_gpt(
    request: GPTGenerateRequest,
):

    try:

        result = await generate_with_gpt(
            prompt=request.prompt,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Third-party API failed: {exc}",
        ) from exc

    return GPTGenerateResponse(
        text=result["text"],
        model=result["model"],
        response_id=result["response_id"],
    )
