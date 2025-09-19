# agent_server.py
import os
import logging
import time
import asyncio
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from openai import OpenAI
from fastapi import HTTPException
from openai import APIError, APIStatusError, RateLimitError, OpenAIError
from fastapi import APIRouter

logger = logging.getLogger("agent_server")
router = APIRouter()

# Optional Redis memory (requires REDIS_URL)
redis = None
REDIS_URL = os.environ.get("REDIS_URL")
if REDIS_URL:
    try:
        import redis as redis_lib
        redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logging.warning("Redis not available: %s", e)
        redis = None

OPENAI_KEY = os.environ.get("OPENAI_KEY")
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_KEY env var required")

openai_client = OpenAI(api_key=OPENAI_KEY)

AGENT_SECRET = os.environ.get("AGENT_KEY")  # Bearer token required if set

app = FastAPI(title="Agent Server", version="0.1")
logger = logging.getLogger("agent_server")
logging.basicConfig(level=logging.INFO)

MEMORY_KEY_PREFIX = "agent:memory:"
MAX_MEMORY_ENTRIES = int(os.environ.get("MAX_MEMORY_ENTRIES", "10"))

def get_memory(convo_id: str):
    if not redis:
        return []
    return redis.lrange(MEMORY_KEY_PREFIX + convo_id, 0, -1)

def push_memory(convo_id: str, entry: str):
    if not redis:
        return
    key = MEMORY_KEY_PREFIX + convo_id
    redis.lpush(key, entry)
    redis.ltrim(key, 0, MAX_MEMORY_ENTRIES - 1)

def check_auth(authorization: Optional[str]):
    if AGENT_SECRET is None:
        return True
    if not authorization:
        return False
    if authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        return token == AGENT_SECRET
    return False

class ReplyRequest(BaseModel):
    convo_id: str
    text: str
    metadata: Optional[dict] = None

class ReplyResponse(BaseModel):
    reply: str
    model: Optional[str] = None
    memory_used: Optional[bool] = False

@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}





# existing helper: check_auth, get_memory, push_memory, ReplyRequest, ReplyResponse, etc.
# make sure openai_client is instantiated earlier in the module

@router.post("/api/reply", response_model=ReplyResponse)
async def api_reply(req: ReplyRequest, request: Request, authorization: Optional[str] = Header(None)):
    if not check_auth(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    convo_id = req.convo_id
    user_text = req.text
    memory_items = get_memory(convo_id)

    # Build chat messages
    messages = []
    if memory_items:
        messages.append({"role": "system", "content": "Memory (most recent first):\n" + "\n".join(memory_items)})
    system_prompt = os.environ.get("SYSTEM_PROMPT")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    model = os.environ.get("AGENT_MODEL", "gpt-5-nano-2025-08-07")

    # Preferred numeric settings (read from env but keep optional)
    # Note: we will translate AGENT_MAX_TOKENS -> max_completion_tokens for modern models
    max_tokens_env = os.environ.get("AGENT_MAX_TOKENS")
    temperature_env = os.environ.get("AGENT_TEMPERATURE")

    # Build base kwargs for OpenAI call
    base_kwargs = {
        "model": model,
        "messages": messages,
    }

    if max_tokens_env:
        try:
            base_kwargs["max_completion_tokens"] = int(max_tokens_env)
        except ValueError:
            logger.warning("AGENT_MAX_TOKENS not an int: %s", max_tokens_env)

    if temperature_env is not None:
        try:
            # don't blindly set if empty string
            base_kwargs["temperature"] = float(temperature_env)
        except (ValueError, TypeError):
            logger.warning("AGENT_TEMPERATURE invalid: %s", temperature_env)

    # Attempt the request with graceful fallback(s)
    loop = asyncio.get_event_loop()

    def do_request(kwargs):
        # wrapper for running the blocking client in a threadpool
        return openai_client.chat.completions.create(**kwargs)

    # Try sequence: first attempt with base_kwargs; on failure try fallback modifications.
    attempt = 0
    last_exc = None
    max_attempts = 2

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info("Calling OpenAI (attempt %d) model=%s", attempt, model)
            resp = await loop.run_in_executor(None, lambda: do_request(base_kwargs.copy()))
            # success — extract message robustly
            choice_text = None
            if hasattr(resp, "choices") and len(resp.choices) > 0:
                choice = resp.choices[0]
                msg = getattr(choice, "message", None)
                if msg is not None:
                    choice_text = getattr(msg, "content", None)
                else:
                    choice_text = getattr(choice, "text", None)
            elif isinstance(resp, dict):
                ch = resp.get("choices", [])
                if ch:
                    c0 = ch[0]
                    if "message" in c0:
                        choice_text = c0["message"].get("content")
                    else:
                        choice_text = c0.get("text")
            if not choice_text:
                logger.error("No reply from OpenAI (empty choices): %s", str(resp)[:500])
                raise HTTPException(status_code=502, detail="No reply from agent")
            # write simple memory (best-effort)
            try:
                push_memory(convo_id, f"user: {user_text}")
                push_memory(convo_id, f"assistant: {choice_text}")
            except Exception as ex:
                logger.warning("Memory write failed: %s", ex)
            # Return result
            return ReplyResponse(reply=choice_text, model=model, memory_used=bool(memory_items))

        except OpenAIError as e:
            last_exc = e
            err_str = str(e)
            logger.warning("OpenAIError on attempt %d: %s", attempt, err_str)

            # Analyze the message to drive fallback decisions:
            # - If the error mentions 'max_tokens' unsupported, remove any max_... token arg
            # - If the error complains about temperature unsupported, remove or set to 1.0
            # - Otherwise, try one fallback then abort

            modified = False
            # handle unsupported parameter: max_tokens
            if "max_tokens" in err_str or "max_completion_tokens" in err_str:
                if "max_completion_tokens" in base_kwargs:
                    logger.info("Removing max_completion_tokens due to OpenAI error")
                    base_kwargs.pop("max_completion_tokens", None)
                    modified = True
            # handle unsupported temperature value
            if "temperature" in err_str and ("does not support" in err_str or "unsupported" in err_str):
                # Some models only accept default 1.0 — try forcing 1.0
                if "temperature" in base_kwargs and base_kwargs["temperature"] != 1.0:
                    logger.info("Resetting temperature to 1.0 due to OpenAI error")
                    base_kwargs["temperature"] = 1.0
                    modified = True
                else:
                    # if no temperature or already 1.0, remove temperature arg
                    base_kwargs.pop("temperature", None)
                    modified = True

            if not modified:
                # If we couldn't identify a safe fallback, stop retrying
                logger.exception("OpenAI call failed: %s", err_str)
                break

            # else, loop and retry
            logger.info("Retrying OpenAI call with modified args (attempt %d)", attempt + 1)

        except Exception as e:
            # Non-OpenAI errors (network, etc.)
            last_exc = e
            logger.exception("Unexpected error calling OpenAI")
            break

    # If we exit loop without returning, everything failed
    logger.exception("OpenAI call failed (all attempts). Last error: %s", str(last_exc)[:1000] if last_exc else "none")
    raise HTTPException(status_code=502, detail="Agent failed to produce reply")
