"""vLLM-based async LLM inference server.

This FastAPI server provides high-performance LLM inference using vLLM's async engine.
Features:
- Continuous batching for efficient GPU utilization
- Streaming and non-streaming generation
- Guided decoding with JSON schemas
- Throughput tracking (requests per second/minute)
- Chat template support for instruction-tuned models

Endpoints:
- POST /generate: Text generation with optional chat formatting
- GET /health: Health check
- GET /stats: Performance statistics
"""

import os, time, asyncio, uuid, json, logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from collections import deque
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from loguru import logger

# =========================
# Config
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "VECTORinc/UrduLLM-4B-Distilled")
print(f"Using model: {MODEL_NAME}")
DTYPE = os.getenv("DTYPE", "bfloat16")

MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEM_UTIL", "0.95"))
CPU_OFFLOAD_GB = float(os.getenv("CPU_OFFLOAD_GB", "0"))

# KV_CACHE_BYTES = int(os.getenv("KV_CACHE_BYTES", None))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "600"))
USE_CHAT_TEMPLATE_ENV = os.getenv("USE_CHAT_TEMPLATE", "").strip().lower()

app = FastAPI(title="vLLM Async server (continuous batching)")
engine: Optional[AsyncLLMEngine] = None
tokenizer: Optional[AutoTokenizer] = None
hf_cfg: Optional[AutoConfig] = None

# =========================
# Logging setup
# =========================
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# Separate logger for max length errors
os.makedirs("llm_logs", exist_ok=True)
max_len_logger = logger.bind(name="max_len_errors")
max_len_logger.add(
    "llm_logs/max_length_errors.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="WARNING",
    rotation="100 MB",
)

# =========================
# Throughput tracking
# =========================
START_TIMES = deque()
RPS_MAX = 0
RPM_MAX = 0
RATE_LOCK: Optional[asyncio.Lock] = None


def _update_rates_on_start(now: float) -> Dict[str, float]:
    """Update and return request rate statistics.
    
    Tracks requests per second (RPS) and requests per minute (RPM) using
    a sliding window approach. Maintains peak values for monitoring.
    
    Args:
        now: Current timestamp (from time.time())
        
    Returns:
        Dictionary containing:
        - rps_current: Current requests per second
        - rpm_current: Current requests per minute
        - rps_max: Peak RPS observed
        - rpm_max: Peak RPM observed
        
    Note:
        Thread-safe when used with RATE_LOCK
    """
    global RPS_MAX, RPM_MAX
    while START_TIMES and (now - START_TIMES[0]) > 60.0:
        START_TIMES.popleft()
    START_TIMES.append(now)
    rps_current = sum(1 for t in START_TIMES if (now - t) <= 1.0)
    rpm_current = len(START_TIMES)
    if rps_current > RPS_MAX:
        RPS_MAX = rps_current
    if rpm_current > RPM_MAX:
        RPM_MAX = rpm_current
    return {
        "rps_current": rps_current,
        "rpm_current": rpm_current,
        "rps_max": RPS_MAX,
        "rpm_max": RPM_MAX,
    }

# =========================
# Pydantic models
# =========================
class GenerateIn(BaseModel):
    prompts: Union[str, List[str]]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    n: int = 1
    stop: Optional[List[str]] = None
    chat: Optional[bool] = None
    system: Optional[str] = None
    ignore_eos: Optional[bool] = None
    logprobs: Optional[int] = 0
    stream: Optional[bool] = True

    assistant_prefix: Optional[str] = None

    guided_json: Optional[dict] = None
    guided_json_schema: Optional[dict] = Field(default=None)
    json_object: Optional[bool] = Field(default=None)

class GenerateOut(BaseModel):
    num_requests: int
    server_meta: dict
    timings: dict
    totals: dict
    results: List[dict]

# =========================
# Helpers
# =========================
def _parse_env_bool(x: str) -> Optional[bool]:
    """Parse a string environment variable to a boolean."""
    x = x.lower()
    if x in ("1", "true", "yes", "on"):
        return True
    if x in ("0", "false", "no", "off"):
        return False
    return None

def _should_apply_chat_template(model_name: str, chat_flag: Optional[bool]) -> bool:
    """Determine if chat template should be applied."""
    if chat_flag is not None:
        return chat_flag
    env = _parse_env_bool(USE_CHAT_TEMPLATE_ENV) if USE_CHAT_TEMPLATE_ENV else None
    if env is not None:
        return env
    return ("instruct" in model_name.lower()) or ("chat" in model_name.lower())

def _format_with_chat_template(raw: str, system: Optional[str]) -> str:
    """Format the prompt with chat template."""
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": raw})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def _check_prompt_length(prompt: str, req_id: str, original_input: str, system: Optional[str]) -> None:
    """Check if prompt exceeds max model length and log if it does.
    
    Args:
        prompt: The formatted prompt to check
        req_id: Request ID for tracking
        original_input: The original user input
        system: The system message (if any)
    """
    try:
        # Tokenize the prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        token_count = len(prompt_tokens)
        token_percentage = round((token_count / MAX_MODEL_LEN) * 100, 2)
        
        if token_count > MAX_MODEL_LEN:
            error_msg = (
                f"PROMPT EXCEEDS MAX LENGTH:\n"
                f"  Request ID: {req_id}\n"
                f"  Token Count: {token_count} / {MAX_MODEL_LEN} ({token_percentage}%)\n"
                f"  System Message: {system if system else 'None'}\n"
                f"  User Input: {original_input}\n"
                f"  Formatted Prompt Length (chars): {len(prompt)}"
            )
            max_len_logger.warning(error_msg)
            logger.warning(f"[REQ {req_id}] PROMPT EXCEEDS MAX LENGTH: {token_count} / {MAX_MODEL_LEN} ({token_percentage}%)")
        else:
            # Also log when within limits for reference
            logger.info(f"[REQ {req_id}] Prompt tokens: {token_count} / {MAX_MODEL_LEN} ({token_percentage}%)")
    except Exception as e:
        logger.error(f"[REQ {req_id}] Error checking prompt length: {e}")


# =========================
# Startup
# =========================
@app.on_event("startup")
async def _startup() -> None:
    """Initialize the LLM engine and tokenizer on startup."""
    global engine, tokenizer, hf_cfg, RATE_LOCK
    RATE_LOCK = asyncio.Lock()

    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "/media/vector/Seagate-2TB/RBT/models/UrduLLM-4B-Distilled",
    )

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=True, trust_remote_code=True
    )

    args = AsyncEngineArgs(
        model=MODEL_PATH,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        gpu_memory_utilization=GPU_MEM_UTIL,
        # cpu_offload_gb=CPU_OFFLOAD_GB,
        disable_log_stats=False,
        tensor_parallel_size=int(os.getenv("TP_SIZE", "1")),
        pipeline_parallel_size=int(os.getenv("PP_SIZE", "1")),
    )

    engine = AsyncLLMEngine.from_engine_args(args)

    try:
        await engine.check_health()
    except Exception:
        pass

# =========================
# Health
# =========================
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_NAME, "max_model_len": MAX_MODEL_LEN}

# =========================
# Generate endpoint
# =========================
@app.post("/generate")
async def generate(body: GenerateIn, request: Request):
    """Generate text using the LLM model."""
    raws = [body.prompts] if isinstance(body.prompts, str) else list(body.prompts)
    if len(raws) != 1:
        raise HTTPException(400, "Send exactly one prompt per request")
    raw = raws[0]

    use_chat = _should_apply_chat_template(MODEL_NAME, body.chat)

    # Log what's being sent to the LLM
    logger.info(f"[REQ] INPUT → user_content: {raw!r}")
    if body.system:
        logger.info(f"[REQ ] INPUT → system: {body.system!r}")
    try:
        final_prompt = (
            _format_with_chat_template(raw, body.system) if use_chat else raw
        )
    except Exception as e:
        raise HTTPException(400, f"chat template failed: {e}")

    # If an assistant_prefix is provided, append it so the model continues from it
    if body.assistant_prefix:
        final_prompt = final_prompt + body.assistant_prefix

    # Generate request ID first so we can use it for logging
    req_id = str(uuid.uuid4())
    
    # Check if prompt exceeds max length and log it
    _check_prompt_length(final_prompt, req_id, raw, body.system)

    # Guided decoding params
    guided = None

    # 1) If backend sends a JSON SCHEMA (your MQG case), pass it straight to vLLM
    if body.guided_json_schema is not None:
        guided = StructuredOutputsParams(json=body.guided_json_schema)
    elif body.guided_json is not None:
        guided = StructuredOutputsParams(json=body.guided_json)
    elif body.json_object:
        guided = StructuredOutputsParams(json_object=True)

    sp = SamplingParams(
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        n=body.n,
        stop=body.stop,
        ignore_eos=bool(body.ignore_eos) if body.ignore_eos is not None else False,
        logprobs=int(body.logprobs or 0),
        prompt_logprobs=None,
        structured_outputs=guided,
    )

    t_req = time.perf_counter()
    t_first: Optional[float] = None

    # STREAMING PATH
    if body.stream:
        async def sse_events():
            nonlocal t_first
            last_count: Dict[int, int] = {}
            full_json_parts: List[str] = []  # <--- COLLECT ONLY SCHEMA RESPONSES

            try:
                yield ":" + (" " * 2048) + "\n\n"
                now = time.perf_counter()
                async with RATE_LOCK:
                    rates = _update_rates_on_start(now)

                meta = {
                    "model_name": MODEL_NAME,
                    "dtype": DTYPE,
                    "max_model_len": MAX_MODEL_LEN,
                    "chat_template_active": bool(use_chat),
                    "structured": bool(guided is not None),
                }
                yield f"data: {json.dumps({'event':'started','req_id':req_id,'server_meta':meta,'rates':rates})}\n\n"

                async for out in engine.generate(final_prompt, sp, req_id):

                    if await request.is_disconnected():
                        await engine.abort(req_id)
                        yield f"data: {json.dumps({'event':'aborted','req_id':req_id})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    emitted = False
                    for i, o in enumerate(out.outputs or []):
                        tids = list(getattr(o, 'token_ids', []) or [])
                        start = max(0, last_count.get(i, 0))

                        for j in range(start, len(tids)):
                            if t_first is None:
                                t_first = time.perf_counter()
                                ttft_ms = (t_first - t_req) * 1000.0
                                yield f"data: {json.dumps({'event':'ttft','ms':round(ttft_ms,3),'req_id':req_id})}\n\n"

                            tid = int(tids[j])
                            tok = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)

                            # === CAPTURE SCHEMA OUTPUT ONLY ===
                            if guided is not None:
                                full_json_parts.append(tok)

                            yield f"data: {json.dumps({'event':'token','index':i,'seq_id':j,'token_id':tid,'text':tok}, ensure_ascii=False)}\n\n"
                            emitted = True

                        last_count[i] = len(tids)

                    if emitted:
                        yield ":\n\n"

                    if getattr(out, "finished", False):
                        # ===== FINISHED — PRINT ONLY SCHEMA JSON =====
                        if guided is not None:
                            full_json_text = ''.join(full_json_parts)
                            logger.error(
                                f"\n=== [REQ {req_id}] GUIDED-JSON STREAM OUTPUT ===\n{full_json_text}\n=============================\n"
                            )

                        t_done = time.perf_counter()
                        if t_first is None:
                            t_first = t_done

                        queue_ms = (t_first - t_req) * 1000.0
                        gen_ms = (t_done - t_first) * 1000.0
                        total_prompt_toks = len(getattr(out, "prompt_token_ids", []) or [])
                        total_out_toks = sum(len(getattr(o, "token_ids", []) or []) for o in out.outputs or [])
                        total_tokens = total_prompt_toks + total_out_toks
                        token_ratio = round((total_tokens / MAX_MODEL_LEN) * 100, 2)

                        logger.info(
                            f"[REQ {req_id}] prompt={total_prompt_toks} out={total_out_toks} "
                            f"→ total={total_tokens}/{MAX_MODEL_LEN} ({token_ratio}%)"
                        )

                        timings = {
                            "queue_ms": round(queue_ms,3),
                            "generate_ms": round(gen_ms,3),
                            "total_ms": round(queue_ms + gen_ms,3)
                        }
                        totals = {
                            "num_requests":1,
                            "total_prompt_tokens": total_prompt_toks,
                            "total_output_tokens": total_out_toks
                        }

                        yield f"data: {json.dumps({'event':'completed','timings':timings,'totals':totals,'req_id':req_id})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

            except asyncio.CancelledError:
                try: await engine.abort(req_id)
                except Exception: pass
                yield f"data: {json.dumps({'event':'cancelled','req_id':req_id})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                try: await engine.abort(req_id)
                except Exception: pass
                yield f"data: {json.dumps({'event':'error','message':str(e),'req_id':req_id})}\n\n"
                yield "data: [DONE]\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(sse_events(), headers=headers, media_type="text/event-stream")

    # NON-STREAMING PATH
    final_output = None
    t_first = None
    try:
        async for out in engine.generate(final_prompt, sp, req_id):
            if await request.is_disconnected():
                await engine.abort(req_id)
                raise HTTPException(499, "Client disconnected")
            # Capture TTFT: first iteration where any output token has been produced
            if t_first is None and out.outputs and any(
                getattr(o, "token_ids", None) for o in out.outputs
            ):
                t_first = time.perf_counter()
            if getattr(out, "finished", False):
                final_output = out
                break

    except asyncio.CancelledError:
        await engine.abort(req_id)
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await engine.abort(req_id)
        except Exception:
            pass
        raise HTTPException(500, f"vLLM error: {e}")

    if final_output is None:
        raise HTTPException(500, "Engine returned no final output")

    t_done = time.perf_counter()
    total_prompt_toks = len(getattr(final_output, "prompt_token_ids", []) or [])

    outs, total_out_toks = [], 0
    for o in final_output.outputs:
        tok_count = len(getattr(o, "token_ids", []) or [])
        total_out_toks += tok_count
        outs.append({
            "text": o.text,
            "tokens": tok_count,
            "finish_reason": getattr(o, "finish_reason", None)
        })

    # === PRINT RAW JSON OUTPUT IF GUIDED JSON IS ENABLED ===
    if guided is not None:
        try:
            json_text = "\n".join(o["text"] for o in outs)
            logger.info(
                f"\n=== [REQ {req_id}] GUIDED-JSON NON-STREAM OUTPUT ===\n{json_text}\n=============================\n"
            )
        except Exception as e:
            logger.error(f"[REQ {req_id}] Failed to log JSON output: {e}")

    total_tokens = total_prompt_toks + total_out_toks
    token_ratio = round((total_tokens / MAX_MODEL_LEN) * 100, 2)

    logger.info(
        f"[REQ {req_id}] prompt={total_prompt_toks} out={total_out_toks} "
        f"→ total={total_tokens}/{MAX_MODEL_LEN} ({token_ratio}%)"
    )

    server_meta = {
        "model_name": MODEL_NAME,
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "chat_template_active": bool(use_chat),
        #"kv_cache_bytes_configured": KV_CACHE_BYTES,
        "engine": "AsyncLLMEngine",
        "structured": bool(guided is not None),
    }

    timings = {
        "queue_ms": round((t_first - t_req) * 1000.0, 3) if t_first else None,
        "generate_ms": round((t_done - t_first) * 1000.0, 3) if t_first else None,
        "total_ms": round((t_done - t_req) * 1000.0, 3)
    }
    totals = {
        "num_requests": 1,
        "total_prompt_tokens": total_prompt_toks,
        "total_output_tokens": total_out_toks,
        "tps_decode": None
    }

    return GenerateOut(
        num_requests=1,
        server_meta=server_meta,
        timings=timings,
        totals=totals,
        results=[{
            "prompt": getattr(final_output, "prompt", final_prompt),
            "prompt_tokens": total_prompt_toks,
            "outputs": outs,
        }],
    )

if __name__ == "__main__":
    uvicorn.run("llm_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), workers=1)
