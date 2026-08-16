import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_llm.llm import LLM
from agent_llm import SamplingParams
from agent_llm.engine.sequence import Sequence


# ============================================================
# 1. nano-vLLM 配置
# ============================================================

MODEL_PATH = "/home/tjwei/llms/Qwen3-0.6B/"

MAX_MODEL_LEN = 2048

# GPU 每个 decode step 最多同时运行多少条 sequence。
# 4 -> 8 -> 16
MAX_NUM_SEQS = 8

# 一次 prefill 最多处理多少 token。
MAX_NUM_BATCHED_TOKENS = 4096

GPU_MEMORY_UTILIZATION = 0.85


# HTTP 层最多允许多少请求排队。
MAX_QUEUE_SIZE = 64

# 最多允许多少请求进入 nano-vLLM scheduler。
MAX_ACTIVE_REQUESTS = 32


# ============================================================
# 2. 请求 / 响应模型
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
        le=1024,
    )

    ignore_eos: bool = False


class GenerateResponse(BaseModel):
    text: str
    output_tokens: int


# ============================================================
# 3. 内部 Job
# ============================================================

@dataclass
class InferenceJob:
    request: GenerateRequest
    future: asyncio.Future


# ============================================================
# 4. 全局状态
# ============================================================

llm: LLM | None = None

request_queue: asyncio.Queue | None = None

engine_task: asyncio.Task | None = None

engine_executor: ThreadPoolExecutor | None = None

engine_error: str | None = None


stats = {
    "active_requests": 0,
    "completed_requests": 0,

    "engine_steps": 0,

    "last_decode_batch_size": 0,
    "max_observed_decode_batch_size": 0,
}


# ============================================================
# 5. 创建 nano-vLLM
# LLM 的创建和之后的推理全部放在同一个 worker thread 中。
# FastAPI event loop 不直接执行 CUDA 推理。
# ============================================================

def create_llm() -> LLM:
    print("Loading nano-vLLM model...")

    model = LLM(
        MODEL_PATH,

        enforce_eager=False,

        tensor_parallel_size=1,

        max_model_len=MAX_MODEL_LEN,

        max_num_seqs=MAX_NUM_SEQS,

        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,

        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    print("nano-vLLM model loaded.")

    return model


# ============================================================
# 6. 把 HTTP 请求加入 nano-vLLM scheduler
#
# 为什么这里不用 llm.generate()？
#
# generate() 会：
#
#     add_request()
#     while not finished:
#         step()
#
# 它会一直占着控制权直到这一批请求全部结束。
#
# 现在自己控制：
#
#     add request
#     step
#     add new request
#     step
#     step
#     ...
#
# 这样新 HTTP 请求可以在旧请求 decode 的过程中插入。
# ============================================================

def engine_add_requests(
    requests: list[tuple[str, SamplingParams]],
):
    assert llm is not None

    results = []

    for prompt, sampling_params in requests:
        try:
            # tokenize
            token_ids = llm.tokenizer.encode(prompt)

            if not token_ids:
                raise ValueError(
                    "Prompt produced zero tokens"
                )

            # 防止 prompt + completion 超过 context window
            if (
                len(token_ids)
                + sampling_params.max_tokens
                > MAX_MODEL_LEN
            ):
                raise ValueError(
                    f"prompt_tokens({len(token_ids)}) "
                    f"+ max_tokens({sampling_params.max_tokens}) "
                    f"> max_model_len({MAX_MODEL_LEN})"
                )

            # nano-vLLM 原来的 add_request()
            # 本质也是构造 Sequence 然后 scheduler.add()
            #
            # 我们这里自己构造，是为了拿到 seq_id，
            # 后面才能知道完成的是哪个 HTTP 请求。
            seq = Sequence(
                token_ids,
                sampling_params,
            )

            llm.scheduler.add(seq)

            results.append(
                (
                    seq.seq_id,
                    len(token_ids),
                    None,
                )
            )

        except Exception as exc:
            results.append(
                (
                    None,
                    0,
                    str(exc),
                )
            )

    return results


# ============================================================
# 7. 执行 nano-vLLM 的一个 scheduler step
# ============================================================

def engine_step_once():
    assert llm is not None

    outputs, num_tokens = llm.step()

    finished = []

    for seq_id, token_ids in outputs:

        text = llm.tokenizer.decode(
            token_ids
        )

        finished.append(
            (
                seq_id,
                text,
                len(token_ids),
            )
        )

    return finished, num_tokens


# ============================================================
# 8. continuous batching 主循环
# ============================================================

async def continuous_batching_loop():
    global engine_error

    assert request_queue is not None
    assert engine_executor is not None

    loop = asyncio.get_running_loop()

    # seq_id -> HTTP job
    pending: dict[int, InferenceJob] = {}

    try:

        while True:

            # ------------------------------------------------
            # A. 如果现在完全没任务，就阻塞等待第一个请求
            # ------------------------------------------------

            jobs_to_add: list[InferenceJob] = []

            if not pending:

                job = await request_queue.get()

                jobs_to_add.append(job)

            # ------------------------------------------------
            # B. 把当前已经到达的 HTTP 请求尽量取出来
            #
            # 但 scheduler 内最多只保留 MAX_ACTIVE_REQUESTS
            # ------------------------------------------------

            capacity = (
                MAX_ACTIVE_REQUESTS
                - len(pending)
                - len(jobs_to_add)
            )

            while capacity > 0:

                try:
                    job = request_queue.get_nowait()

                except asyncio.QueueEmpty:
                    break

                jobs_to_add.append(job)

                capacity -= 1

            # ------------------------------------------------
            # C. 加入 nano-vLLM scheduler
            # ------------------------------------------------

            if jobs_to_add:

                engine_inputs = []

                for job in jobs_to_add:

                    req = job.request

                    sampling_params = SamplingParams(
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        ignore_eos=req.ignore_eos,
                    )

                    engine_inputs.append(
                        (
                            req.prompt,
                            sampling_params,
                        )
                    )

                add_results = await loop.run_in_executor(
                    engine_executor,
                    engine_add_requests,
                    engine_inputs,
                )

                admitted = 0

                for job, result in zip(
                    jobs_to_add,
                    add_results,
                ):

                    seq_id, prompt_tokens, error = result

                    request_queue.task_done()

                    if error is not None:

                        if not job.future.done():

                            job.future.set_exception(
                                ValueError(error)
                            )

                        continue

                    pending[seq_id] = job

                    admitted += 1

                stats["active_requests"] = len(pending)

                if admitted:

                    print(
                        "[engine] "
                        f"admitted={admitted}, "
                        f"active={len(pending)}, "
                        f"queue={request_queue.qsize()}"
                    )

            # ------------------------------------------------
            # D. 没有 active sequence，就继续等待 HTTP 请求
            # ------------------------------------------------

            if not pending:
                continue

            # ------------------------------------------------
            # E. 只执行 nano-vLLM 一个 step
            #
            # 非常重要：
            #
            # 执行完一个 step 后马上回到循环开头，
            # 因此这期间新来的 HTTP 请求会被加入 scheduler。
            #
            # 这就是 continuous batching 的关键。
            # ------------------------------------------------

            finished, num_tokens = (
                await loop.run_in_executor(
                    engine_executor,
                    engine_step_once,
                )
            )

            stats["engine_steps"] += 1

            # nano-vLLM 当前实现：
            #
            # prefill:
            #     num_tokens > 0
            #
            # decode:
            #     num_tokens = -len(seqs)
            #
            # 所以负数的绝对值就是当前 decode batch size。
            if num_tokens < 0:

                decode_batch_size = -num_tokens

                previous = stats[
                    "last_decode_batch_size"
                ]

                stats[
                    "last_decode_batch_size"
                ] = decode_batch_size

                stats[
                    "max_observed_decode_batch_size"
                ] = max(
                    stats[
                        "max_observed_decode_batch_size"
                    ],
                    decode_batch_size,
                )

                # batch size 变化时打印一次，
                # 避免每个 token 都疯狂刷屏。
                if decode_batch_size != previous:

                    print(
                        "[engine] "
                        f"decode_batch="
                        f"{decode_batch_size}, "
                        f"active="
                        f"{len(pending)}, "
                        f"queue="
                        f"{request_queue.qsize()}"
                    )

            # ------------------------------------------------
            # F. 某些 sequence 已经生成完成
            # ------------------------------------------------

            for (
                seq_id,
                text,
                output_tokens,
            ) in finished:

                job = pending.pop(
                    seq_id,
                    None,
                )

                if job is None:
                    continue

                if not job.future.done():

                    job.future.set_result(
                        GenerateResponse(
                            text=text,
                            output_tokens=output_tokens,
                        )
                    )

                stats["completed_requests"] += 1

            stats["active_requests"] = len(pending)

    except asyncio.CancelledError:
        raise

    except Exception as exc:

        engine_error = str(exc)

        print(
            f"[engine] fatal error: {exc}"
        )

        # 已经进入 scheduler 的请求全部失败
        for job in pending.values():

            if not job.future.done():

                job.future.set_exception(
                    RuntimeError(
                        f"Inference engine failed: {exc}"
                    )
                )

        pending.clear()

        stats["active_requests"] = 0

        # queue 里还没有进入 engine 的也返回错误
        while True:

            try:
                job = request_queue.get_nowait()

            except asyncio.QueueEmpty:
                break

            request_queue.task_done()

            if not job.future.done():

                job.future.set_exception(
                    RuntimeError(
                        f"Inference engine failed: {exc}"
                    )
                )


# ============================================================
# 9. FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global llm
    global request_queue
    global engine_task
    global engine_executor
    global engine_error

    engine_error = None

    request_queue = asyncio.Queue(
        maxsize=MAX_QUEUE_SIZE
    )

    # 只有一个 nano-vLLM engine thread
    engine_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="nano-vllm-engine",
    )

    loop = asyncio.get_running_loop()

    # 模型加载也放到这个 thread
    llm = await loop.run_in_executor(
        engine_executor,
        create_llm,
    )

    # 开 continuous batching loop
    engine_task = asyncio.create_task(
        continuous_batching_loop()
    )

    print(
        "Continuous batching engine started."
    )

    yield

    print(
        "FastAPI server shutting down..."
    )

    if engine_task is not None:

        engine_task.cancel()

        try:
            await engine_task

        except asyncio.CancelledError:
            pass

    if engine_executor is not None:

        engine_executor.shutdown(
            wait=True,
            cancel_futures=True,
        )


# ============================================================
# 10. FastAPI
# ============================================================

app = FastAPI(
    title="nano-vLLM Continuous Batching Server",
    version="0.2.0",
    lifespan=lifespan,
)


# ============================================================
# 11. health
# ============================================================

@app.get("/health")
async def health():

    queue_size = 0

    if request_queue is not None:
        queue_size = request_queue.qsize()

    return {
        "status": (
            "ok"
            if engine_error is None
            else "error"
        ),

        "model_loaded": (
            llm is not None
        ),

        "engine_error": engine_error,

        "queue_size": queue_size,

        "active_requests": stats[
            "active_requests"
        ],

        "completed_requests": stats[
            "completed_requests"
        ],

        "engine_steps": stats[
            "engine_steps"
        ],

        "max_num_seqs": MAX_NUM_SEQS,

        "last_decode_batch_size": stats[
            "last_decode_batch_size"
        ],

        "max_observed_decode_batch_size": stats[
            "max_observed_decode_batch_size"
        ],
    }


# ============================================================
# 12. generate
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(
    request: GenerateRequest,
):

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if engine_error is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference engine unavailable: "
                f"{engine_error}"
            ),
        )

    if request_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Request queue is not ready",
        )

    loop = asyncio.get_running_loop()

    future = loop.create_future()

    job = InferenceJob(
        request=request,
        future=future,
    )

    # 不无限排队。
    try:
        request_queue.put_nowait(job)

    except asyncio.QueueFull:

        raise HTTPException(
            status_code=429,
            detail=(
                "Inference queue is full. "
                "Please retry later."
            ),
        )

    try:

        result = await future

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc
