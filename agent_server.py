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
from openai.error import OpenAIError
import asyncio
import os

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



@app.post("/api/reply", response_model=ReplyResponse)
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
    max_tokens_env = os.environ.get("AGENT_MAX_TOKENS", "512")
    try:
        max_completion_tokens = int(max_tokens_env)
    except Exception:
        max_completion_tokens = 512

    temperature = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))

    try:
        # run blocking client in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # use the newer param name expected by modern OpenAI endpoints
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                ),
            )
        except OpenAIError as oe:
            # If the model/SDK combination still expects `max_tokens` (rare), try fallback once
            msg = str(oe)
            logger.warning("OpenAIError on first attempt: %s", msg)
            # try a fallback call without the newer param (older API)
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_completion_tokens,
                        temperature=temperature,
                    ),
                )
            except OpenAIError as oe2:
                # log both error messages for diagnosis and re-raise for outer handler
                logger.exception("OpenAI call failed (fallback attempt also failed): %s | %s", oe, oe2)
                raise

        # extract text robustly from response (support dict or object shapes)
        choice_text = None

        # openai-python may return resource objects or plain dicts depending on version
        # Try common locations in order of likelihood
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
            # new-style: choice.message.content
            msg = getattr(choice, "message", None)
            if msg is not None:
                choice_text = getattr(msg, "content", None)
            # older-style: choice.text
            if not choice_text:
                choice_text = getattr(choice, "text", None)
            # delta streaming style
            if not choice_text:
                delta = getattr(choice, "delta", None)
                if isinstance(delta, dict):
                    choice_text = delta.get("content")
        elif isinstance(resp, dict):
            ch = resp.get("choices", [])
            if ch:
                c0 = ch[0]
                if isinstance(c0.get("message"), dict):
                    choice_text = c0["message"].get("content")
                else:
                    choice_text = c0.get("text") or (c0.get("delta") or {}).get("content")

        if not choice_text:
            # Log enough of the response to debug without dumping huge payloads
            logger.error("No reply from OpenAI. Response (truncated): %s", str(resp)[:2000])
            raise HTTPException(status_code=502, detail="No reply from agent (empty response)")

        # write simple memory (best-effort)
        try:
            push_memory(convo_id, f"user: {user_text}")
            push_memory(convo_id, f"assistant: {choice_text}")
        except Exception as ex:
            logger.warning("Memory write failed: %s", ex)

        # return the field names matching your ReplyResponse model
        return ReplyResponse(reply=choice_text, model=model, memory_used=bool(memory_items))

    except OpenAIError as e:
        # extract helpful diagnostic text, but avoid leaking secrets
        err_text = getattr(e, "user_message", None) or str(e)
        logger.exception("OpenAI call failed: %s", err_text)
        # Return 502 with concise message so caller sees cause
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {err_text}")

    except HTTPException:
        # re-raise HTTPExceptions we intentionally raised above
        raise

    except Exception as e:
        logger.exception("Unexpected error in api_reply: %s", e)
        raise HTTPException(status_code=502, detail="Agent failed to produce reply")
