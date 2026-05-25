from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from mangum import Mangum
from pydantic import BaseModel
from openai import OpenAI
import anthropic
from anthropic import Anthropic
from google import genai
from dotenv import load_dotenv
import os
import io
import json
import queue
import threading
import concurrent.futures
import time
import asyncio
import urllib.request
import urllib.parse
import hmac
import base64
from google.genai import types

from datetime import datetime, timezone
from typing import List, Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_root, '.env'))

app = FastAPI()

DAILY_LIMIT = 15          # per-IP daily query limit (general mode)
INTENSIVE_LIMIT = 3       # per-IP daily query limit (intensive/coding mode)
_UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
_UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")


MODELS_GENERAL = {
    "gpt":    "gpt-5.4-mini-2026-03-17",
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3.5-flash",
    "grok":   "grok-4-1-fast-non-reasoning",
}
MODELS_INTENSIVE = {
    "gpt":    "gpt-5.4-2026-03-05",
    "claude": "claude-opus-4-7",
    "gemini": "gemini-3.1-pro-preview",
    # Grok is not available in intensive mode
}


def _redis(command: list) -> object:
    """Call Upstash Redis REST API. Returns parsed JSON result field."""
    if not _UPSTASH_URL or not _UPSTASH_TOKEN:
        return None
    path = "/".join(urllib.parse.quote(str(c), safe="") for c in command)
    req = urllib.request.Request(
        f"{_UPSTASH_URL}/{path}",
        headers={"Authorization": f"Bearer {_UPSTASH_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())["result"]


FEEDBACK_LIMIT = 3          # max submissions per IP per hour
FEEDBACK_WINDOW = 3600      # seconds

def check_feedback_rate_limit(ip: str) -> bool:
    """Returns True if the request is allowed, False if the IP is over the limit."""
    try:
        hour = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H')
        key = f"council:feedback:{ip}:{hour}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, FEEDBACK_WINDOW])
        return count <= FEEDBACK_LIMIT
    except Exception:
        return True  # fail open — an Upstash outage shouldn't block feedback


def _extract_user_id(request: "Request") -> str | None:
    """Decode the Supabase JWT from the Authorization header and return the user sub.
    No signature verification — used only as a stable rate-limit key."""
    import base64 as _b64
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(_b64.urlsafe_b64decode(padded))
        return payload.get("sub") or None
    except Exception:
        return None


def _client_id(request: "Request") -> str:
    """Return user:<uuid> for authenticated requests, ip:<addr> otherwise."""
    uid = _extract_user_id(request)
    if uid:
        return f"user:{uid}"
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return f"ip:{ip}"


def check_and_increment_ip_limit(request: "Request") -> tuple[bool, int]:
    """Returns (allowed, current_count) keyed by user ID (if authed) or IP. Resets every 24h."""
    try:
        key = f"council:prompts:{_client_id(request)}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, 86400])
        return (count <= DAILY_LIMIT, int(count))
    except Exception:
        return (True, 0)  # fail open so an Upstash outage doesn't break the site


def check_and_increment_intensive_limit(request: "Request") -> tuple[bool, int]:
    """Returns (allowed, current_count) for intensive queries. Separate 3/day bucket."""
    try:
        key = f"council:intensive:{_client_id(request)}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, 86400])
        return (count <= INTENSIVE_LIMIT, int(count))
    except Exception:
        return (True, 0)



@app.get("/assets/{filename:path}")
def serve_asset(filename: str):
    path = os.path.join(_root, "templates", "assets", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
perplexity_client = OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

# ── Synthesis system prompts (edit prompts/*.txt to tune behavior) ───────────
_PROMPTS_DIR = os.path.join(_root, "prompts")

def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read().strip()

SYNTHESIS_SYSTEM_TEXT  = _load_prompt("synthesis_text.txt")
SYNTHESIS_SYSTEM_IMAGE = _load_prompt("synthesis_image.txt")

# ── Cost table (USD per 1M tokens) ───────────────────────────────────────────
# Update input/output prices here when provider pricing changes.
_PRICING = {
    "claude-haiku-4-5-20251001":      {"in": 0.80,  "out": 4.00},
    "claude-sonnet-4-6":              {"in": 3.00,  "out": 15.00},
    # Fill in below when pricing is confirmed for these models
    "gpt-5.2-2025-12-11":             {"in": 0.0,   "out": 0.0},
    "gemini-3.1-pro-preview":         {"in": 0.0,   "out": 0.0},
    "gemini-3.1-flash-image-preview": {"in": 0.0,   "out": 0.0},
    "grok-4.20-0309-non-reasoning":   {"in": 0.0,   "out": 0.0},
}

def _cost(model: str, input_tok: int, output_tok: int) -> float:
    p = _PRICING.get(model, {"in": 0.0, "out": 0.0})
    return (input_tok * p["in"] + output_tok * p["out"]) / 1_000_000

_LOG_KEY  = "council:logs"
_LOG_MAX  = 1000   # keep the most recent N entries in Redis
_LOG_PATH = os.path.join(_root, "logs", "queries.jsonl")  # local fallback
_log_lock = threading.Lock()


def _redis_post(command: list) -> object:
    """POST a command to Upstash (better for large values than the GET path API)."""
    if not _UPSTASH_URL or not _UPSTASH_TOKEN:
        return None
    data = json.dumps(command).encode()
    req = urllib.request.Request(
        _UPSTASH_URL,
        data=data,
        headers={"Authorization": f"Bearer {_UPSTASH_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())["result"]


def _write_log(entry: dict):
    line = json.dumps(entry)
    # Write to Redis if available (works in production)
    try:
        if _UPSTASH_URL and _UPSTASH_TOKEN:
            _redis_post(["RPUSH", _LOG_KEY, line])
            _redis(["LTRIM", _LOG_KEY, -_LOG_MAX, -1])
            return
    except Exception as e:
        print(f"[LOG] Redis write failed, falling back to file: {e}", flush=True)
    # Local file fallback (skipped on read-only filesystems like Vercel)
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with _log_lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


def truncate(text, max_chars=5000):
    if not text:
        return "No response"
    return text[:max_chars] + "..." if len(text) > max_chars else text


def estimate_max_tokens(question):
    q = question.lower()
    words = len(question.split())
    heavy_keywords = [
        'write', 'implement', 'code', 'build', 'create', 'essay', 'script',
        'program', 'function', 'algorithm', 'step by step', 'in detail',
        'comprehensive', 'full', 'complete', 'entire', 'tutorial',
    ]
    detailed_keywords = [
        'explain', 'compare', 'analyse', 'analyze', 'difference', 'pros and cons',
        'advantages', 'disadvantages', 'how does', 'why does', 'describe',
        'elaborate', 'overview', 'summarize', 'list', 'examples',
    ]
    oneword_keywords = ['yes or no', 'true or false', 'how many', 'how much']
    if any(kw in q for kw in heavy_keywords) or words > 60:
        return 4000
    if any(kw in q for kw in detailed_keywords) or words > 25:
        return 2500
    if any(kw in q for kw in oneword_keywords):
        return 800
    return 1500


# ── Council model functions ───────────────────────────────────────────────────

def ask_gpt(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["gpt"]
    messages = list(history or []) + [{"role": "user", "content": question}]
    try:
        response = openai_client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=messages,
        )
    except Exception as e:
        err = str(e)
        print(f"[GPT] {model} failed with max_completion_tokens: {err}", flush=True)
        # Some models reject max_completion_tokens — retry without it
        if "max_completion_tokens" in err or "unknown" in err.lower() or "unsupported" in err.lower():
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
        else:
            raise
    usage = {"model": model,
             "input_tokens": response.usage.prompt_tokens,
             "output_tokens": response.usage.completion_tokens}
    content = response.choices[0].message.content
    if not content:
        content = (getattr(response.choices[0].message, 'refusal', None)
                   or getattr(response.choices[0].message, 'reasoning_content', None)
                   or "")
    return content, usage


def ask_claude(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["claude"]
    messages = list(history or []) + [{"role": "user", "content": question}]
    for attempt in range(4):
        try:
            with anthropic_client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            ) as stream:
                text = stream.get_final_text()
                u = stream.get_final_message().usage
                usage = {"model": model,
                         "input_tokens": u.input_tokens,
                         "output_tokens": u.output_tokens}
                return text, usage
        except anthropic.InternalServerError as e:
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("Claude is currently overloaded — please try again in a moment.") from e


def ask_gemini(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["gemini"]
    contents = []
    for msg in (history or []):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    # question may be a string or a list of Gemini parts (for multimodal)
    if isinstance(question, list):
        contents.append({"role": "user", "parts": question})
    else:
        contents.append({"role": "user", "parts": [{"text": question}]})
    response = gemini_client.models.generate_content(
        model=model,
        config={"maxOutputTokens": max_tokens},
        contents=contents,
    )
    meta = response.usage_metadata
    usage = {"model": model,
             "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
             "output_tokens": getattr(meta, "candidates_token_count", 0) or 0}
    try:
        text = response.text
    except ValueError:
        # Response was blocked or cut off — extract text from parts directly
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        text = "".join(p.text for p in parts if hasattr(p, "text") and p.text)
    return text, usage


def ask_grok(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["grok"]
    messages = list(history or []) + [{"role": "user", "content": question}]
    response = grok_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    usage = {"model": model,
             "input_tokens": response.usage.prompt_tokens,
             "output_tokens": response.usage.completion_tokens}
    return response.choices[0].message.content, usage


# ── Synthesis generators ──────────────────────────────────────────────────────

def synthesize_with_claude(prompt, max_tokens, history=None, system=None, model=None):
    model = model or MODELS_GENERAL["claude"]
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    for attempt in range(4):
        try:
            with anthropic_client.messages.stream(**kwargs) as stream_obj:
                for chunk in stream_obj.text_stream:
                    yield chunk
                final = stream_obj.get_final_message()
                print(f"[SYNTH Claude] stop_reason={final.stop_reason}  output_tokens={final.usage.output_tokens}  max_tokens={max_tokens}", flush=True)
            return
        except anthropic.InternalServerError as e:
            if attempt < 3:
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("Claude is currently overloaded — please try again in a moment.") from e


def synthesize_with_gpt(prompt, max_tokens, history=None, system=None, model=None):
    model = model or MODELS_GENERAL["gpt"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages += list(history or []) + [{"role": "user", "content": prompt}]
    try:
        stream = openai_client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=messages,
            stream=True,
        )
    except Exception as e:
        err = str(e)
        print(f"[GPT synth] {model} max_completion_tokens={max_tokens} failed: {err}", flush=True)
        if "max_completion_tokens" in err or "unknown" in err.lower() or "unsupported" in err.lower():
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
        else:
            raise
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def synthesize_with_gemini(prompt, max_tokens, history=None, system=None, model=None):
    model = model or MODELS_GENERAL["gemini"]
    contents = []
    for msg in (history or []):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    config = {"maxOutputTokens": max_tokens}
    if system:
        config["system_instruction"] = system
    def _stream(cfg):
        for chunk in gemini_client.models.generate_content_stream(
            model=model, config=cfg, contents=contents,
        ):
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                candidate = chunk.candidates[0] if chunk.candidates else None
                parts = candidate.content.parts if candidate and candidate.content else []
                text = "".join(p.text for p in parts if hasattr(p, "text") and p.text)
                if text:
                    yield text
    try:
        yield from _stream(config)
    except Exception as e:
        err = str(e)
        print(f"[Gemini synth] maxOutputTokens={max_tokens} failed: {err} — retrying without cap", flush=True)
        if "maxOutputTokens" in err or "invalid" in err.lower() or "exceed" in err.lower():
            fallback_config = {k: v for k, v in config.items() if k != "maxOutputTokens"}
            yield from _stream(fallback_config)
        else:
            raise


def synthesize_with_grok(prompt, max_tokens, history=None, system=None, model=None):
    model = model or MODELS_GENERAL["grok"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages += list(history or []) + [{"role": "user", "content": prompt}]
    stream = grok_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Image generation helpers ──────────────────────────────────────────────────

def enrich_prompt_for_imagen(question, responses):
    """Builds a precise Imagen prompt from question + text-model creative direction."""
    parts = [f"Original request: {question}"]
    for name, text in responses.items():
        if text:
            parts.append(f"{name}'s creative direction: {truncate(text, 800)}")
    combined = "\n\n".join(parts)
    meta_prompt = (
        f"{combined}\n\nWrite ONE precise image generation prompt for Google Imagen. "
        "Include: subject, composition, art style, color palette, lighting, mood, and "
        "quality specs. Output ONLY the prompt text, no preamble."
    )
    try:
        with anthropic_client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            messages=[{"role": "user", "content": meta_prompt}],
        ) as s:
            return s.get_final_text().strip()
    except Exception:
        return question


def call_gemini_imagen(prompt):
    """Calls Gemini image generation 4 times in parallel, returns base64-encoded image strings."""
    def generate_one(_):
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return base64.b64encode(part.inline_data.data).decode("utf-8")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(generate_one, i) for i in range(4)]
        images = [f.result() for f in futures]
    return [img for img in images if img]


# ── Request models ────────────────────────────────────────────────────────────

class Attachment(BaseModel):
    type: str  # 'text' | 'image'
    content: str
    filename: str

class StreamRequest(BaseModel):
    question: str = ""
    synthesizer: str = "claude"
    models: List[str] = ["gpt", "claude", "gemini", "grok"]
    history: List[dict] = []
    image_mode: bool = False
    attachments: List[Attachment] = []
    mode: str = "general"  # "general" or "intensive"
    web_search: bool = False
    artifact_mode: bool = False
    forced_artifact_type: str = ""  # "code" | "research" | "" (auto-detect)
    research_depth: str = "standard"  # "quick" | "standard" | "deep"
    prior_code: str = ""          # current build's code, sent on Studio Build follow-ups
    studio_followup: bool = False # True when iterating on an existing build artifact

class ResearchRequest(BaseModel):
    topic: str = ""
    context: str = ""
    follow_up_of: str = ""        # previous topic, when this is a follow-up exploration
    history: List[dict] = []

class FeedbackRequest(BaseModel):
    message: str = ""
    sentiment: str = ""

class AdminVerifyRequest(BaseModel):
    code: str = ""


# ── Perplexity Sonar (web-aware council member) ───────────────────────────────

def synthesize_with_perplexity(prompt: str, max_tokens: int, search_query: str = ""):
    messages = []
    if search_query:
        messages.append({"role": "system", "content": (
            f"The user's original question is: {search_query}\n"
            f"Your web search MUST be focused specifically on this question. "
            f"Only retrieve sources directly relevant to this question."
        )})
    messages.append({"role": "user", "content": prompt})
    response = perplexity_client.chat.completions.create(
        model="sonar",
        max_tokens=max_tokens,
        messages=messages,
    )
    content = response.choices[0].message.content
    citations = getattr(response, 'citations', None)
    if citations is None and hasattr(response, 'model_extra') and response.model_extra:
        citations = response.model_extra.get('citations')
    citations = list(citations) if citations else []
    return content, citations


# ── Attachment helpers ────────────────────────────────────────────────────────

def build_attachment_context(attachments: list) -> str:
    if not attachments:
        return ""
    lines = ["The user has attached the following file(s) for context:\n"]
    for att in attachments:
        lines.append(f"--- {att.filename} ---")
        lines.append(att.content)
        lines.append("---\n")
    return "\n".join(lines)


# ── Artifact mode helpers ─────────────────────────────────────────────────────

_ARTIFACT_KEYWORDS = (
    'compare', 'contrast', 'explain', 'write', 'create', 'outline',
    'analyze', 'analyse', 'summarize', 'summarise', 'guide', 'tutorial',
    'essay', 'plan', 'design', 'strategy', 'framework', 'breakdown', 'research',
)


def should_create_artifact(question: str) -> bool:
    q = (question or "").lower()
    return any(kw in q for kw in _ARTIFACT_KEYWORDS)


_CODE_ACTIONS = ('build', 'create', 'make', 'generate', 'develop', 'code', 'implement', 'design')
_CODE_TARGETS = (
    'website', 'web app', 'webapp', 'application', 'app', 'landing page', 'page',
    'dashboard', 'tool', 'game', 'calculator', 'component', 'ui', 'interface',
    'portfolio', 'blog', 'form', 'site', 'frontend', 'prototype',
)


def detect_artifact_type(question: str) -> str:
    q = (question or "").lower()
    # Code generation: action verb + product noun, or "build/create/make me a …"
    if (any(a in q for a in _CODE_ACTIONS) and any(t in q for t in _CODE_TARGETS)):
        return "code"
    if any(phrase in q for phrase in ('build me', 'create me', 'make me', 'generate me', 'code me')):
        return "code"
    if any(w in q for w in ('compare', ' vs ', 'versus', 'difference')):
        return "comparison"
    if any(w in q for w in ('explain', 'how does', 'what is', 'what are')):
        return "explanation"
    if any(w in q for w in ('essay', 'article')) or q.startswith('write'):
        return "essay"
    if any(w in q for w in ('guide', 'tutorial', 'how to')):
        return "guide"
    if any(w in q for w in ('analyze', 'analyse', 'breakdown', 'analysis')):
        return "analysis"
    return "research"


def suggest_artifact_name(question: str, artifact_type: str) -> str:
    words = (question or "").split()
    key_words = [w.strip(".,!?;:'\"") for w in words if len(w) > 4][:3]
    suggestion = " ".join(key_words).title()
    return suggestion if suggestion else f"Untitled {artifact_type}"


# ── Build follow-up intent (question vs edit) ─────────────────────────────────
# Hybrid: a fast keyword heuristic decides clear cases; an LLM resolves ambiguous ones.

_EDIT_SIGNALS = (
    'add ', 'change', 'make it', 'make the', 'make them', 'remove', 'delete', 'replace',
    'instead', 'refactor', 'fix ', 'update', 'modify', 'turn it into', 'turn into',
    'support ', 'enable', 'include', 'give it', 'set the', 'set it', 'increase', 'decrease',
    'rename', 'resize', 'recolor', 're-color', 'restyle', 'implement', 'integrate', 'connect',
    'animate', 'center', 'align', 'bigger', 'smaller', 'darker', 'lighter', 'faster', 'slower',
    'also add', 'can you add', 'could you add', 'please add', 'put a', 'put an', 'let the user',
    'allow the user', 'should be able', 'make sure', 'get rid of', 'swap', 'convert it',
)
_QUESTION_STARTS = (
    'what', 'why', 'how', 'when', 'where', 'which', 'who', 'is ', 'are ', 'does ', 'do ',
    'can you explain', 'could you explain', 'explain', 'tell me', 'is there', 'are there',
    'should i', "what's", 'whats', 'do i', 'will it', 'would it',
)


def _heuristic_intent(message: str):
    """Return 'edit', 'question', or None when the phrasing is ambiguous."""
    m = (message or "").strip().lower()
    if not m:
        return 'question'
    has_edit = any(s in m for s in _EDIT_SIGNALS)
    has_question = m.endswith('?') or any(m.startswith(s) for s in _QUESTION_STARTS)
    if has_edit and not has_question:
        return 'edit'
    if has_question and not has_edit:
        return 'question'
    return None


def _classify_intent_with_model(message: str) -> str:
    try:
        prompt = (
            "A user is iterating on a code project. Classify their follow-up message as exactly one word:\n"
            "EDIT = they want the code changed, extended, added to, fixed, or restyled.\n"
            "QUESTION = they are only asking or inquiring and do NOT want the code changed.\n\n"
            f"Follow-up: \"{message}\"\n\n"
            "Answer with one word only: EDIT or QUESTION."
        )
        out, _ = ask_gpt(prompt, 4, model=MODELS_GENERAL.get('gpt'))
        return 'edit' if 'edit' in (out or '').lower() else 'question'
    except Exception:
        # If unsure, prefer answering over silently rebuilding the user's app.
        return 'question'


def classify_followup_intent(message: str) -> str:
    h = _heuristic_intent(message)
    return h if h is not None else _classify_intent_with_model(message)


def search_web_for_artifact(question: str) -> list:
    """Run Perplexity to gather live sources for an artifact. Returns list of {url,title,content}."""
    try:
        prompt = (
            f"Research this topic and provide a concise, factual brief with citations: {question}\n\n"
            "Pull the most authoritative and current sources. Cite each major claim with [1], [2], etc."
        )
        content, citations = synthesize_with_perplexity(prompt, 1500, search_query=question)
    except Exception as e:
        print(f"[ARTIFACT] web search failed: {e}", flush=True)
        return []
    sources = []
    summary = (content or "").strip()
    for i, url in enumerate((citations or [])[:8]):
        domain = ""
        try:
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
        except Exception:
            pass
        sources.append({
            "url": url,
            "title": domain or url,
            "content": summary[:400] if i == 0 else "",
        })
    return sources


_DEPTH_TARGETS = {"quick": 6, "standard": 10, "deep": 18}


def _check_doi_valid(doi: str) -> bool:
    """Returns True if the DOI resolves via CrossRef. On network errors, assumes valid."""
    if not doi:
        return True
    plain = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').strip()
    if not plain or plain.startswith('http'):
        return True  # plain URL, not a DOI — skip
    url = f'https://api.crossref.org/works/{urllib.parse.quote(plain, safe="")}'
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Council/1.0 (mailto:support@opclave.com)')
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code not in (404, 410)
    except Exception:
        return True  # timeout or network error — don't penalise


def _verify_paper_dois(papers: list) -> list:
    """Verify all paper DOIs in parallel; clear any that CrossRef says don't exist."""
    if not papers:
        return papers
    def _check(p):
        doi = (p.get('doi') or '').strip()
        if doi and not _check_doi_valid(doi):
            return {**p, 'doi': ''}
        return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(papers), 8)) as ex:
        return list(ex.map(_check, papers))


def search_for_research_papers(question: str, depth: str = "standard") -> list:
    """
    Use Perplexity to find specific peer-reviewed papers, returning structured metadata.
    Returns a list of dicts: {authors, year, title, journal, doi, finding}.
    """
    target = _DEPTH_TARGETS.get(depth, 10)
    search_prompt = (
        "You are searching academic databases for peer-reviewed research.\n\n"
        f"Question: {question}\n\n"
        f"Find up to {target} specific papers that directly address different angles of this question.\n\n"
        "SOURCE QUALITY RULES — strictly enforced:\n"
        "ONLY return sources from the following categories:\n"
        "  1. Peer-reviewed papers published in academic journals or conferences relevant to "
        "the question's domain. Examples by field (not exhaustive):\n"
        "     - General/multidisciplinary: Science, Nature, PNAS, PLOS ONE, Cell\n"
        "     - Medicine/health: The Lancet, NEJM, JAMA, BMJ, Nature Medicine\n"
        "     - Psychology/neuroscience: Psychological Science, Nature Neuroscience, Neuron\n"
        "     - Economics/finance: American Economic Review, QJE, Journal of Finance, NBER\n"
        "     - Computer science/AI: arXiv, NeurIPS, ICML, ICLR, CVPR, ACL, IEEE, ACM\n"
        "     - Climate/environment: Nature Climate Change, GRL, Global Environmental Change\n"
        "     - Social science/policy: World Development, American Political Science Review\n"
        "  2. Authoritative institutional reports — first-tier sources for policy, economics, "
        "public health, and geopolitical questions:\n"
        "     - UN agencies: WHO, FAO, UNHCR, UNDP, UNEP, UNICEF, ILO, UNESCO\n"
        "     - Financial institutions: World Bank, IMF (Working Papers, WEO, Fiscal Monitor)\n"
        "     - Energy/climate: IEA, IRENA, IPCC Assessment Reports and Special Reports\n"
        "     - OECD Economic Surveys, Health at a Glance, Education at a Glance\n"
        "     - Central bank research papers (Federal Reserve, ECB, BIS working papers)\n"
        "     - Government statistics offices (Census Bureau, ONS, Eurostat, etc.)\n"
        "  3. Official technical reports from major research institutions with primary empirical "
        "data (RAND, Brookings research with data, peer-reviewed think-tank working papers)\n\n"
        "DO NOT return any of the following — exclude entirely:\n"
        "  - Company blogs or marketing content\n"
        "  - Personal blogs, Substack posts, or Medium articles\n"
        "  - YouTube videos or video content of any kind\n"
        "  - Course overview pages or syllabi\n"
        "  - Library research guides (e.g., LibGuides)\n"
        "  - General explainer websites (Wikipedia, Investopedia, Towards Data Science)\n"
        "  - News articles without a direct link to a primary research source\n"
        "  - Think-tank opinion pieces without primary empirical data\n\n"
        "Prioritise among qualifying sources:\n"
        "1. Empirical studies and reports with real quantitative data, projections with confidence intervals\n"
        "2. Meta-analyses, systematic reviews, and multi-study assessments (e.g., IPCC syntheses)\n"
        "3. Sources that challenge or complicate the mainstream consensus\n"
        "4. Recent publications (2020-2026) that may update or contradict older findings\n"
        "5. Sources from different disciplines that bring different methodological approaches\n"
        "6. Sources that provide specific regional or comparative data (not only global aggregates)\n\n"
        "If you cannot find peer-reviewed sources for a claim or sub-question, "
        "include an entry with title 'NO PEER-REVIEWED SOURCE FOUND' and finding set to "
        "'No qualifying peer-reviewed source was found for: <the specific claim>'. "
        "Do NOT substitute a blog post, explainer, or non-peer-reviewed source.\n\n"
        "For EACH qualifying paper return: Author(s), Year, Title, Journal/Conference, DOI or URL, "
        "and a one-sentence factual finding from the paper.\n\n"
        "Output ONLY a JSON array (no prose, no code fences) shaped like:\n"
        '[{"authors":"Vosoughi et al.","year":2018,"title":"...","journal":"Science",'
        '"doi":"10.1126/science.aap9559","finding":"..."}]\n\n'
        "CRITICAL: Only include papers you are confident actually exist. If unsure, exclude it. "
        "If the finding is unclear, write 'finding unclear from abstract' rather than inventing one."
    )
    try:
        response = perplexity_client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": search_prompt}],
            max_tokens=2500,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        print(f"[RESEARCH] Perplexity paper search failed: {e}", flush=True)
        return []

    # Strip code fences if Perplexity wrapped the JSON
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Remove a leading "json" tag if present
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    # Find the first '[' and last ']' as a fallback
    if not text.startswith("["):
        first = text.find("[")
        last = text.rfind("]")
        if first != -1 and last > first:
            text = text[first:last + 1]
    try:
        papers = json.loads(text)
        if not isinstance(papers, list):
            return []
        # Normalise each entry
        out = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            out.append({
                "authors": str(p.get("authors", "") or "").strip(),
                "year":    p.get("year") or "",
                "title":   str(p.get("title", "") or "").strip(),
                "journal": str(p.get("journal", "") or "").strip(),
                "doi":     str(p.get("doi", "") or "").strip(),
                "finding": str(p.get("finding", "") or "").strip(),
            })
        return out
    except Exception as e:
        print(f"[RESEARCH] paper JSON parse failed: {e}", flush=True)
        return []


def format_papers_for_prompt(papers: list) -> str:
    """Format the paper list as a numbered citation block for model/synthesis prompts."""
    if not papers:
        return ""
    lines = []
    for i, p in enumerate(papers):
        authors = p.get("authors") or "Unknown"
        year = p.get("year") or "n.d."
        title = p.get("title") or "Untitled"
        journal = p.get("journal") or ""
        doi = p.get("doi") or ""
        finding = p.get("finding") or ""
        meta = f"{authors} ({year}). \"{title}\""
        if journal:
            meta += f". {journal}"
        if doi:
            meta += f" (DOI: {doi})"
        line = f"[{i+1}] {meta}"
        if finding:
            line += f"\n     Finding: {finding}"
        lines.append(line)
    return "\n".join(lines)


def verify_artifact_citations(synthesis_text: str, papers: list) -> tuple:
    """
    Check that every [N] citation maps to a paper in the list and flag unsupported claims.
    Returns (is_valid, errors).
    """
    import re as _re
    errors = []
    n_papers = len(papers)
    cites = _re.findall(r'\[(\d+)\]', synthesis_text or "")
    for c in cites:
        try:
            num = int(c)
        except ValueError:
            continue
        if num < 1 or num > n_papers:
            errors.append(f"Citation [{num}] is out of range (only {n_papers} papers available).")

    # Flag unqualified claims that don't have a citation within ~80 chars
    unqualified = [
        r'\bresearch shows\b',
        r'\bstudies (?:show|find|indicate)\b',
        r'\bit is (?:known|established)\b',
        r'\bexperts agree\b',
        r'\bevidence shows\b',
    ]
    for pat in unqualified:
        for m in _re.finditer(pat, synthesis_text or "", _re.IGNORECASE):
            window = (synthesis_text or "")[m.start():m.end() + 80]
            if not _re.search(r'\[\d+\]', window):
                errors.append(f"Unqualified claim near \"{window[:48]}…\" — add citation or rephrase.")
                break  # one warning per pattern is enough

    return (len(errors) == 0, errors)


def format_sources_for_context(sources: list) -> str:
    if not sources:
        return ""
    lines = ["Live web research (cite as [1], [2], etc when you use these facts):"]
    for i, s in enumerate(sources):
        title = s.get("title", "") or ""
        url = s.get("url", "") or ""
        content = (s.get("content") or "")[:300]
        suffix = f" — {content}" if content else ""
        lines.append(f"[{i+1}] {title} ({url}){suffix}")
    return "\n".join(lines)


# Research artifact: each model answers the FULL question from a distinct lens.
# These are treated as raw research inputs — the synthesizer extracts substance,
# not model opinions. The reader never sees which model said what.
_ARTIFACT_ROLES_RESEARCH = {
    "GPT": (
        "You are a domain expert producing a rigorous research input on the topic given. "
        "Your mandate is quantitative empirical grounding.\n\n"
        "SCOPE NOTE (write 2-3 sentences at the top): Define what time horizon, geographic scope, "
        "and evidence base you are drawing from for this specific topic. Name the primary datasets, "
        "report series, or authoritative sources that are the foundation of your input "
        "(e.g., for neuroscience: key journals and meta-analyses; for economics: central bank or "
        "IMF datasets; for climate: IPCC reports; for public health: WHO or Lancet studies). "
        "This prevents the output from appearing as cherry-picked anecdotes.\n\n"
        "CONTENT REQUIREMENTS:\n"
        "- Every major claim must carry a quantitative anchor where empirical data exists for this "
        "topic. Replace vague statements with specific figures: not 'outcomes improved' but "
        "'outcomes improved by X% over Y years in population Z (source)'. If no quantitative data "
        "exists for a claim, say so explicitly.\n"
        "- Distinguish clearly between: (a) scientific consensus supported by multiple independent "
        "studies, (b) findings from individual studies, (c) projections or models with stated "
        "confidence intervals or uncertainty ranges, and (d) genuine open questions. Label each.\n"
        "- Name the authoritative source for each key figure. Specify the report, journal, edition, "
        "or institutional body — not just the institution name alone.\n"
        "- Cover the primary mechanisms and established evidence at the depth required for someone "
        "building on this for their own research or decision-making.\n\n"
        "Aim for 600-800 words of authoritative, citation-anchored prose."
    ),
    "Claude": (
        "You are a domain expert producing a rigorous research input on the topic given. "
        "Your mandate is critical analysis and counter-argument engagement.\n\n"
        "CONTENT REQUIREMENTS:\n"
        "- Identify the strongest counter-arguments and objections to the mainstream view on this "
        "topic. Do not dismiss them — steelman them. If the dominant framework has a known "
        "weakness, name it explicitly and explain what it would take to resolve it.\n"
        "- Distinguish what the evidence actually supports from what is commonly claimed. Point to "
        "specific cases where the dominant narrative overstates, oversimplifies, or misrepresents "
        "the underlying findings.\n"
        "- Address high-impact, low-probability risk scenarios (tail risks) relevant to this topic "
        "as primary analytical variables, not as peripheral footnotes. For each: describe the "
        "plausible mechanism, assess the potential magnitude, and explain why mainstream analysis "
        "tends to underweight it (short-term incentives, measurement difficulty, etc.).\n"
        "- Identify the assumptions that underlie the mainstream position on this topic. What would "
        "have to be true for the dominant view to be wrong? What evidence would falsify it?\n"
        "- Operationalize any heuristic or conceptual shorthand terms relevant to this topic. If "
        "you introduce or use a coined label, define the specific measurable criteria by which "
        "something qualifies — do not leave it as an impressionistic phrase.\n"
        "- Acknowledge genuine uncertainty honestly. If the evidence is mixed or the literature is "
        "sparse, say so rather than projecting false confidence.\n\n"
        "Aim for 500-700 words. Intellectual honesty over comprehensiveness."
    ),
    "Gemini": (
        "You are a domain expert producing a rigorous research input on the topic given. "
        "Your mandate is empirical comparative analysis.\n\n"
        "CONTENT REQUIREMENTS:\n"
        "- Ground every claim in a specific named example, data point, case study, or mechanism. "
        "Generic statements are unacceptable. Replace 'outcomes vary' with a concrete named "
        "instance of that variation — who, where, by how much, and according to which study.\n"
        "- Provide at least one comparative analysis: contrast two cases, actors, countries, "
        "institutions, or approaches that faced the same challenge with different methods or "
        "starting conditions. Explain what the comparison reveals that a single case would not.\n"
        "- Where the topic involves policy, regulation, or institutional mechanisms, name the "
        "specific instruments and explain how they function — not just that they exist. Trace the "
        "mechanism from design intent to real-world effect, including any documented gaps.\n"
        "- Quantify impacts wherever empirical data exists. Name the figure, the source, and any "
        "important methodology caveats (sample size, time period, geographic scope).\n"
        "- Identify who is concretely affected, how, and what the downstream consequences are for "
        "practitioners, policymakers, or the populations directly involved.\n\n"
        "Aim for 500-700 words. Every paragraph must have a concrete empirical anchor."
    ),
    "Grok": (
        "You are a domain expert producing a rigorous research input on the topic given. "
        "Your mandate is the research frontier, structural gaps, and conceptual operationalization.\n\n"
        "CONTENT REQUIREMENTS:\n"
        "- Surface the active debates researchers are currently having in this specific field. "
        "Be specific: name the competing positions, the evidence cited on each side, and what "
        "makes the debate genuinely unresolved rather than merely controversial.\n"
        "- Identify structural or systemic dynamics that standard analyses of this topic tend to "
        "miss or underweight. This might include second-order effects, perverse incentives, "
        "interdependencies across systems, or the gap between stated rationale and actual "
        "mechanisms in relevant policies or institutions.\n"
        "- Treat high-impact risk scenarios as primary variables with their appropriate analytical "
        "weight — not as footnotes. For each relevant tail-risk scenario: what is the plausible "
        "trigger, what is the realistic magnitude of impact, and what makes it structurally "
        "underrepresented in conventional analysis of this topic?\n"
        "- Operationalize any conceptual shorthand or coined terms with specific measurable "
        "criteria. A useful heuristic label is only useful if you can say precisely when something "
        "qualifies and when it does not.\n"
        "- Flag recent findings (2021-2026) that challenge established views in this field, "
        "and explain why they matter.\n\n"
        "Aim for 500-600 words. Prioritize specificity and intellectual honesty over breadth."
    ),
}

# Code artifact: each model acts as a specialist giving creative/technical direction
_ARTIFACT_ROLES_CODE = {
    "GPT": (
        "You are a software architect. Your job is creative and technical direction ONLY — do NOT write code.\n"
        "Cover: overall architecture, recommended tech stack, file structure, key design decisions and trade-offs.\n"
        "Be specific and opinionated. 300-500 words."
    ),
    "Claude": (
        "You are a UX and frontend engineer. Your job is creative and technical direction ONLY — do NOT write code.\n"
        "Cover: component breakdown, user flows, visual design principles, interactions, and accessibility.\n"
        "Be specific and opinionated. 300-500 words."
    ),
    "Gemini": (
        "You are a performance and data engineer. Your job is creative and technical direction ONLY — do NOT write code.\n"
        "Cover: data models, animations/visual effects worth including, performance considerations, "
        "and any libraries or CDN resources that would elevate the result.\n"
        "Be specific and opinionated. 300-500 words."
    ),
    "Grok": (
        "You are a creative director. Your job is creative direction ONLY — do NOT write code.\n"
        "Cover: visual identity, colour palette, typography, unique features that would make this stand out, "
        "and any unexpected angles worth exploring.\n"
        "Be specific and opinionated. 300-400 words."
    ),
}


def build_artifact_model_prompt(model_name: str, question: str, web_context: str,
                                 artifact_type: str = "", papers: list = None,
                                 prior_code: str = "") -> str:
    if artifact_type == "code":
        role = _ARTIFACT_ROLES_CODE.get(model_name, "Provide creative direction for this build request. Do NOT write code.")
        if prior_code:
            return (
                f"{role}\n\n"
                "IMPORTANT: This is an iteration on an EXISTING, working app — not a new build. "
                "The user wants a focused change. Preserve everything else.\n\n"
                f"Current code:\n```\n{truncate(prior_code, 6000)}\n```\n\n"
                f"Requested change: {question}\n\n"
                "Give concise, specific direction for implementing ONLY this change while keeping all "
                "existing functionality, structure, and styling intact. Do NOT write the full code."
            )
        parts = [role, "", f"Build request: {question}"]
        return "\n".join(parts)

    role = _ARTIFACT_ROLES_RESEARCH.get(model_name, "Answer the following question thoroughly.")

    # If we have a paper list, force source-grounded analysis
    if papers:
        papers_text = format_papers_for_prompt(papers)
        return (
            f"{role}\n\n"
            f"Question: {question}\n\n"
            f"Below are {len(papers)} peer-reviewed papers retrieved for this question. "
            "These are your ONLY allowed sources for factual claims:\n\n"
            f"{papers_text}\n\n"
            "RULES:\n"
            "1. Synthesise what THESE papers actually found. Do not speculate or use general knowledge to fill gaps.\n"
            "2. Use [1], [2], [3] citations corresponding to the paper numbers above EVERY TIME you reference a finding.\n"
            "3. If a paper's finding is unclear from what is provided, say so explicitly — do not invent.\n"
            "4. Mark claims with their support level:\n"
            "   - 'Consensus across multiple studies' if 3+ papers agree\n"
            "   - 'Supported by research [N]' if 1-2 papers support it\n"
            "   - 'Debated in the literature' if papers disagree\n"
            "   - 'Speculative' if it requires inference beyond the papers\n"
            "5. Do NOT cite any paper not in the list above.\n"
            "6. If the papers do not address something, write 'The provided research does not directly address...'\n"
            "7. Organise your response by themes that emerge from the papers, not arbitrary topics.\n\n"
            "Write as if for a literature review. Every factual claim must be traceable to a numbered source."
        )

    # Fallback (no papers): old behaviour with general web context
    parts = [role, ""]
    if web_context:
        parts += [web_context, ""]
    parts.append(f"Topic: {question}")
    return "\n".join(parts)


def get_artifact_synthesis_prompt(question: str, model_responses: dict, web_sources: list,
                                   artifact_type: str = "", papers: list = None,
                                   prior_code: str = "") -> str:
    if artifact_type == "code":
        directions_block = "\n\n---\n\n".join(
            f"### {name}:\n{truncate(r, 3500)}"
            for name, r in model_responses.items() if r
        )
        if prior_code:
            return (
                "You are an expert engineer modifying an EXISTING, working single-file app. "
                "Apply the requested change and preserve everything else exactly as it is.\n"
                "Rules: output the COMPLETE updated self-contained HTML file with all CSS and JS inline. "
                "Keep all existing features, structure, IDs, and styling intact except where the change "
                "strictly requires otherwise. Do NOT rebuild from scratch or drop unrelated features. "
                "No placeholders, no TODOs, no prose — start with ```html and end with ```.\n\n"
                f"Current code (modify this):\n```html\n{prior_code}\n```\n\n"
                f"Requested change: {question}\n\n"
                f"Direction from the council on how to apply it:\n\n{directions_block}\n\n"
                "Output the full updated file now, with the change applied and everything else preserved."
            )
        return (
            "You are an expert engineer. You have received creative and architectural direction from "
            "multiple specialists. Your ONLY job is to write the complete, working code.\n"
            "Rules: output a single self-contained HTML file with all CSS and JS embedded inline. "
            "No placeholders. No TODOs. No prose explanations — only brief inline code comments where logic is non-obvious. "
            "Start your response with ```html and end with ```.\n\n"
            f"Build request: {question}\n\n"
            f"Creative direction from the council:\n\n{directions_block}\n\n"
            "Write the complete implementation now. All CSS in a <style> block. All JS in a <script> block. "
            "Use CDN links for libraries (Tailwind, Three.js, GSAP, Chart.js etc). "
            "Make it visually polished and fully functional."
        )

    # Research / document synthesis — authoritative single-voice format
    responses_block = "\n\n---\n\n".join(
        f"[Research input {i+1}]\n{truncate(text, 5000)}"
        for i, (_, text) in enumerate(model_responses.items()) if text
    )

    # Paper-grounded path: the synthesizer must cite ONLY from the verified paper list
    if papers:
        papers_text = format_papers_for_prompt(papers)
        return (
            "You are creating a research artifact from multiple analyses of peer-reviewed papers "
            "and authoritative institutional reports. Your output will be read by someone using it "
            "as the basis for their own research or writing. It must read like a well-researched "
            "scholarly document by a single authoritative voice, not a summary of opinions.\n\n"
            "CRITICAL RULE: Never mention AI models, which input said what, or that this came from "
            "multiple sources. Extract the substance. Write as one expert. The machinery is invisible.\n\n"
            f"Research question: {question}\n\n"
            f"Verified sources (your ONLY allowed citation pool):\n\n{papers_text}\n\n"
            f"Independent expert analyses (treat as raw material, not as positions to compare):\n\n"
            f"{responses_block}\n\n"
            "=== HOW TO USE THESE INPUTS ===\n"
            "1. Where multiple analyses cite the SAME source for the same claim: high-confidence consensus — present it as such\n"
            "2. Where analyses cite DIFFERENT sources for different conclusions: genuine field debate — frame it as 'researchers debate...'\n"
            "3. Where one analysis makes a claim no other analysis and no source supports: drop it\n"
            "4. If an analysis overstates a source's finding, defer to what the source actually says\n"
            "5. Extract quantitative figures from the inputs and anchor them to their sources\n\n"
            "=== OUTPUT FORMAT — LaTeX ===\n\n"
            "Output a complete LaTeX document. Start with \\documentclass{article} and "
            "\\begin{document}. End with \\end{document}. Use \\section{} / \\section*{} "
            "for headings, \\textbf{} for bold, \\textit{} for italics, "
            "\\begin{itemize}/\\begin{enumerate} for lists. "
            "Write inline citations as plain [N] text — do NOT use \\cite{}.\n\n"
            "\\documentclass{article}\n"
            "\\begin{document}\n\n"
            "\\begin{abstract}\n"
            "120-160 words. Direct answer to the question. Must include at least two quantitative "
            "anchors (specific figures with sources). No vague generalities.\n"
            "\\end{abstract}\n\n"
            "\\section*{Methodology Note}\n"
            "80-120 words. Define scope: time horizon, geographic focus, evidence types drawn from. "
            "Name primary datasets or authoritative sources. State explicitly what falls outside scope.\n\n"
            "\\section{[Section title -- a specific dimension of the question]}\n"
            "Write 3--5 substantive sections organised by themes that emerge from the sources.\n"
            "Each section must:\n"
            "\\begin{itemize}\n"
            "\\item Open with a quantified empirical anchor with a citation. "
            "If no quantitative data exists, say so explicitly.\n"
            "\\item Explain underlying mechanisms clearly.\n"
            "\\item Mark confidence: ``Consensus across studies [1][2]'', ``One study suggests [4]'', "
            "``Debated in the literature [5][6]'', or ``The current evidence base does not support...''\n"
            "\\item Present genuine field debates as ``researchers debate...'' -- never ``inputs disagree''.\n"
            "\\item Close with a \\textbf{Counter-evidence:} note (1--3 sentences, strongest objection).\n"
            "\\end{itemize}\n"
            "Each section: 300--500 words.\n\n"
            "\\section*{High-Impact Risk Scenarios}\n"
            "150--250 words. For each scenario: mechanism and trigger, magnitude of impact, "
            "why conventional analysis structurally underweights it. "
            "Cite sources where they exist; acknowledge where the evidence base is thin.\n\n"
            "\\section*{Structural and Policy Dimensions}\n"
            "150--200 words. Structural, institutional, or policy forces shaping the topic. "
            "Name specific mechanisms. Provide at least one comparative example.\n\n"
            "\\section*{Counter-Arguments and Limitations}\n"
            "200--300 words. Engage directly with the strongest objections. Present them fairly. "
            "Address: underlying assumptions, weakest evidence, scope boundaries.\n\n"
            "\\section*{Practical Implications}\n"
            "100--150 words. Concrete, actionable takeaways for policymakers, researchers, practitioners.\n\n"
            "\\section*{Open Questions}\n"
            "80--120 words. Genuinely unresolved issues. Distinguish ``not yet studied'' "
            "from ``studied but inconclusive''.\n\n"
            "\\section*{Sources}\n"
            "List ONLY the sources actually cited above, in the format already provided "
            "(authors, year, title, journal/institution, DOI or URL). "
            "Use \\begin{enumerate}...\\item...\\end{enumerate}.\n\n"
            "\\end{document}\n\n"
            "=== CITATION RULES (follow strictly) ===\n"
            "- Use [N] citations corresponding ONLY to the numbered source list above.\n"
            "- Every factual claim must trace to a specific source [N] OR be explicitly marked "
            "as inference, projection, or speculation.\n"
            "- Describe what each source actually found — do not exceed the 'Finding:' line.\n"
            "- Never cite a source not in the list. Never invent findings, authors, or DOIs.\n"
            "- If a source's finding is unclear, write 'one study suggests...' rather than overstating.\n"
            "- For general claims with no source support, use 'Research suggests...' without [N].\n"
            "- Exclude YouTube links, personal blogs, Substack posts, and non-primary-source "
            "opinion pieces from citations. If such a source appeared in the inputs, drop it.\n\n"
            "=== QUALITY RULES ===\n"
            "- 2500-5000 words total. Never pad. Never truncate substance.\n"
            "- Every paragraph must earn its place: does it change how the reader understands the topic?\n"
            "- Every major claim needs a quantitative anchor where one exists in the literature.\n"
            "- Operationalize all heuristic or coined terms. If you use a conceptual label "
            "(e.g., 'threat multiplier', 'electrostate'), define it with specific measurable criteria "
            "in the text — do not leave it as an impressionistic phrase.\n"
            "- Field debates: 'researchers debate X' — never 'sources differ' or 'models disagree'.\n"
            "- Counter-arguments must be engaged with, not dismissed.\n"
            "- One authoritative voice throughout. Avoid hedging phrases like 'it could be argued' "
            "or 'some might say'.\n"
            "- A reader without the original question should fully understand the topic.\n"
        )

    # Fallback path: no verified papers found, use general web sources
    sources_text = "\n".join(
        f"[{i+1}] {s.get('title','')}: {(s.get('content','') or '')[:200]}".rstrip(": ")
        + (f" ({s.get('url','')})" if s.get('url') else '')
        for i, s in enumerate(web_sources)
    ) or "(no live sources retrieved)"

    return (
        "You are producing a standalone research document from multiple expert research inputs "
        "and live web sources. Your output will be read by someone using it as the basis for "
        "their own research or writing. It must read like a well-researched scholarly document "
        "by a single authoritative voice, not a summary of different opinions.\n\n"
        "CRITICAL RULE: Never mention AI models, which input said what, or that this came from "
        "multiple sources. Extract the substance. Write as one expert. The machinery is invisible.\n\n"
        f"Research question: {question}\n\n"
        f"Expert research inputs (treat as raw material, not as positions to compare):\n\n{responses_block}\n\n"
        f"Live web sources:\n{sources_text}\n\n"
        "=== OUTPUT FORMAT — LaTeX ===\n\n"
        "Output a complete LaTeX document. Start with \\documentclass{article} and "
        "\\begin{document}. End with \\end{document}. Use \\section{} / \\section*{} "
        "for headings, \\textbf{} for bold, \\textit{} for italics, "
        "\\begin{itemize}/\\begin{enumerate} for lists. "
        "Write inline citations as plain [N] text -- do NOT use \\cite{}.\n\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n\n"
        "\\begin{abstract}\n"
        "120-160 words. Direct answer to the question. Include at least two quantitative anchors. "
        "No vague generalities.\n"
        "\\end{abstract}\n\n"
        "\\section*{Methodology Note}\n"
        "80-120 words. Scope, time horizon, geographic focus, evidence types, what falls outside scope.\n\n"
        "\\section{[Section title -- a specific dimension of the question]}\n"
        "Write 3--5 sections of 300--500 words each.\n"
        "Each must open with a quantified anchor, explain mechanisms, present field debates as "
        "``researchers debate...'', and close with \\textbf{Counter-evidence:} (1--3 sentences).\n\n"
        "\\section*{High-Impact Risk Scenarios}\n"
        "150--250 words. Mechanism, magnitude, why underweighted by conventional analysis.\n\n"
        "\\section*{Structural and Policy Dimensions}\n"
        "150--200 words. Structural/institutional/policy forces. One comparative example.\n\n"
        "\\section*{Counter-Arguments and Limitations}\n"
        "200--300 words. Strongest objections, assumptions, weakest evidence, scope boundaries.\n\n"
        "\\section*{Practical Implications}\n"
        "100--150 words. Concrete takeaways. Avoid generic recommendations.\n\n"
        "\\section*{Open Questions}\n"
        "80--120 words. Unresolved issues. Distinguish ``not yet studied'' from ``inconclusive''.\n\n"
        "\\section*{Sources}\n"
        "[1] Authors, Year. Title. Journal/Institution. URL\n\n"
        "\\end{document}\n\n"
        "=== CITATION RULES (follow strictly) ===\n"
        "- Only cite [N] if you are confident about the specific content of that source.\n"
        "- Describe what a study actually found, not what you assume it found.\n"
        "- Never cite by institution name alone — include author, year, and specific finding.\n"
        "- If you cannot verify what a source says, do NOT use [N] — write 'Research suggests...' instead.\n"
        "- Do not fabricate findings from a domain name alone.\n"
        "- Exclude YouTube links, personal blogs, Substack posts, and non-primary-source "
        "opinion pieces. If such a source appears in the inputs, drop it from citations.\n\n"
        "=== QUALITY RULES ===\n"
        "- 2500-5000 words total. Never pad. Never truncate substance.\n"
        "- Every paragraph must earn its place.\n"
        "- Every major claim needs a quantitative anchor where one exists in the literature.\n"
        "- Operationalize all heuristic or coined terms with specific measurable criteria.\n"
        "- Counter-arguments must be engaged with honestly, not dismissed.\n"
        "- Field debates: 'researchers debate X' — never 'sources differ on X'.\n"
        "- One authoritative voice throughout.\n"
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    with open(os.path.join(_root, "templates", "landing.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/studio")
def studio_route():
    with open(os.path.join(_root, "templates", "studio.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/app")
def app_route():
    with open(os.path.join(_root, "templates", "index.html"), encoding="utf-8") as f:
        html_content = f.read()
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    config_script = (
        f'<script>window._SUPABASE_URL="{supabase_url}";'
        f'window._SUPABASE_ANON_KEY="{supabase_anon_key}";</script>'
    )
    html_content = html_content.replace("<!-- __SUPABASE_CONFIG__ -->", config_script)
    return HTMLResponse(html_content)


@app.post("/extract-file")
async def extract_file(file: UploadFile = File(...)):
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTS = {'.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.webp'}
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
    IMAGE_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit")

    if ext in IMAGE_EXTS:
        print(f"[EXTRACT] {filename}: image → Claude vision", flush=True)
        try:
            b64 = base64.b64encode(data).decode("utf-8")
            msg = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": IMAGE_MIME[ext], "data": b64}},
                        {"type": "text", "text": "Describe this image in full detail, including any text, diagrams, equations, charts, or visual content you can see. Be thorough — this description will be used as context by other AI models."},
                    ],
                }],
            )
            return {"type": "image", "content": msg.content[0].text}
        except Exception as e:
            return {"error": f"Image extraction failed: {str(e)}"}

    elif ext == ".pdf":
        print(f"[EXTRACT] {filename}: PDF → local extraction", flush=True)
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return {"type": "text", "content": text}
        except Exception as e:
            return {"error": f"PDF extraction failed: {str(e)}"}

    else:  # .txt, .md
        print(f"[EXTRACT] {filename}: text → local read", flush=True)
        try:
            return {"type": "text", "content": data.decode("utf-8", errors="replace")}
        except Exception as e:
            return {"error": f"Text extraction failed: {str(e)}"}



@app.post("/verify-admin")
def verify_admin(req: AdminVerifyRequest):
    secret = os.getenv("ADMIN_CODE", "")
    if not secret:
        return {"valid": False}
    match = hmac.compare_digest(req.code.encode(), secret.encode())
    return {"valid": match}


@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    message = req.message.strip()
    if not message:
        return {"ok": False, "error": "No message provided"}

    # Rate-limit by IP: max 3 submissions per hour
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    if not check_feedback_rate_limit(ip):
        return {"ok": False, "rate_limited": True, "error": "Too many submissions. Please wait before sending again."}

    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_pass:
        return {"ok": False, "error": "Email not configured"}

    sentiment = req.sentiment.strip()
    if sentiment == "up":
        sentiment_label = "[👍 Positive]"
    elif sentiment == "down":
        sentiment_label = "[👎 Negative]"
    else:
        sentiment_label = ""

    def send_email():
        import smtplib, email.mime.text, email.mime.multipart
        sender = "aarushbajajus@gmail.com"
        recipient = "councilaiclb@gmail.com"
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = f"Opclave — Feedback {sentiment_label}".strip()
        msg.attach(email.mime.text.MIMEText(
            f"Feedback received:\n\n{message}\n\n---\nSent via Opclave", "plain"
        ))
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, gmail_pass)
            server.sendmail(sender, recipient, msg.as_string())

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_email)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/logs")
def get_logs():
    # Try Redis first
    try:
        if _UPSTASH_URL and _UPSTASH_TOKEN:
            raw = _redis(["LRANGE", _LOG_KEY, 0, -1])
            entries = []
            for item in (raw or []):
                try:
                    entries.append(json.loads(item))
                except Exception:
                    pass
            return list(reversed(entries))  # newest first
    except Exception as e:
        print(f"[LOG] Redis read failed: {e}", flush=True)
    # Fall back to local file
    if not os.path.exists(_LOG_PATH):
        return []
    entries = []
    with open(_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(entries))


@app.delete("/logs")
def clear_logs():
    try:
        if _UPSTASH_URL and _UPSTASH_TOKEN:
            _redis(["DEL", _LOG_KEY])
    except Exception:
        pass
    if os.path.exists(_LOG_PATH):
        open(_LOG_PATH, "w").close()
    return {"ok": True}



@app.post("/stream")
def stream(req: StreamRequest, request: Request):
    print(f"[STREAM] received — web_search={req.web_search} question={repr(req.question[:60])}", flush=True)
    question = req.question.strip()
    synthesizer = req.synthesizer.lower()
    requested_models = set(req.models)
    history = req.history
    image_mode = req.image_mode
    attachments = req.attachments
    is_intensive = req.mode.lower() == "intensive"

    # Studio Build follow-up routing: a lightweight question is answered directly (no council),
    # an edit goes through the build pipeline with the current code injected as the base.
    prior_code = (req.prior_code or "").strip()
    is_studio_followup = bool(req.studio_followup) and bool(prior_code)
    followup_intent = classify_followup_intent(question) if is_studio_followup else ""
    if followup_intent == "question":
        # A quick inquiry shouldn't burn a premium credit — bill it as a general query.
        is_intensive = False

    # Enforce intensive-mode restrictions before anything else
    if is_intensive:
        requested_models -= {"grok"}
        if synthesizer == "grok":
            synthesizer = "claude"

    if not question and not attachments:
        def err():
            yield 'data: {"type":"error","message":"No question provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    if is_intensive:
        allowed, count = check_and_increment_intensive_limit(request)
        active_limit = INTENSIVE_LIMIT
    else:
        allowed, count = check_and_increment_ip_limit(request)
        active_limit = DAILY_LIMIT

    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":active_limit,"mode":req.mode})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    attachment_ctx = build_attachment_context(attachments)
    effective_question = question or "Please review the attached file(s)."
    augmented_question = f"{attachment_ctx}\n{effective_question}" if attachment_ctx else effective_question
    artifact_mode = bool(req.artifact_mode) and not image_mode
    do_perplexity = req.web_search and not image_mode and not is_intensive and not artifact_mode

    # ── Build follow-up QUESTION: answer with a single model, no council/synthesis/artifact ──
    if followup_intent == "question":
        q_model_map = MODELS_INTENSIVE if is_intensive else MODELS_GENERAL
        q_synth_map = {
            'claude': synthesize_with_claude, 'gpt': synthesize_with_gpt,
            'gemini': synthesize_with_gemini, 'grok': synthesize_with_grok,
        }
        q_fn = q_synth_map.get(synthesizer, synthesize_with_claude)
        q_model = q_model_map.get(synthesizer) or MODELS_GENERAL.get(synthesizer, MODELS_GENERAL["claude"])
        q_system = (
            "You are a senior software engineer pair-programming with the user on a single-file web app. "
            "Answer their question about the current code clearly and concisely — explain behavior, point to "
            "the relevant parts, and offer options when useful. Do NOT output a full rewrite of the app; "
            "short illustrative snippets are fine only when they genuinely help. Be direct and practical."
        )
        q_prompt = f"Here is the current code:\n\n```html\n{prior_code}\n```\n\n{effective_question}"

        def answer_generate():
            yield f"data: {json.dumps({'type': 'followup_intent', 'intent': 'question'})}\n\n"
            council_text = ""
            try:
                for chunk in q_fn(q_prompt, 4000, history, system=q_system, model=q_model):
                    council_text += chunk
                    yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'council_error', 'error': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        return StreamingResponse(answer_generate(), media_type="text/event-stream")

    def generate():
        nonlocal augmented_question
        artifact_type = ""
        if is_studio_followup:
            yield f"data: {json.dumps({'type': 'followup_intent', 'intent': 'edit'})}\n\n"
        model_tokens = estimate_max_tokens(effective_question)
        if artifact_mode:
            model_tokens = max(model_tokens, 4000)
            synthesis_tokens = 32000
        elif is_intensive:
            model_tokens = max(model_tokens, 4000)
            synthesis_tokens = 32000
        else:
            synthesis_tokens = min(max(model_tokens * 2, 3000), 8000)
        print(f"[DEBUG] Question: {repr(effective_question[:80])}", flush=True)
        print(f"[DEBUG] model_tokens={model_tokens}  synthesis_tokens={synthesis_tokens}", flush=True)
        print(f"[DEBUG] Active models: {requested_models}  Synthesizer: {synthesizer}", flush=True)

        event_queue = queue.Queue()
        results = {}
        usages = {}
        lock = threading.Lock()

        web_sources = []
        papers = []
        if artifact_mode:
            artifact_type = req.forced_artifact_type or detect_artifact_type(effective_question)
            artifact_name = suggest_artifact_name(effective_question, artifact_type)
            if artifact_type == "code":
                synthesis_tokens = 32000
            yield f"data: {json.dumps({'type': 'artifact_mode', 'enabled': True, 'artifact_type': artifact_type, 'suggested_name': artifact_name})}\n\n"

            if artifact_type == "code":
                if not is_studio_followup:
                    # New build: gather web context (frameworks, libs, etc.)
                    yield f"data: {json.dumps({'type': 'searching_web'})}\n\n"
                    web_sources = search_web_for_artifact(effective_question)
                    yield f"data: {json.dumps({'type': 'web_search_complete', 'sources': web_sources})}\n\n"
                # else: iterating on an existing build — skip the web search, we already have the app
            else:
                # Research artifacts: hunt for real peer-reviewed papers
                yield f"data: {json.dumps({'type': 'searching_research', 'depth': req.research_depth})}\n\n"
                papers = _verify_paper_dois(search_for_research_papers(effective_question, req.research_depth))
                yield f"data: {json.dumps({'type': 'papers_found', 'count': len(papers), 'papers': papers})}\n\n"
                if len(papers) < 5:
                    yield f"data: {json.dumps({'type': 'insufficient_research', 'count': len(papers), 'message': f'Only found {len(papers)} relevant papers — falling back to general web research.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'searching_web'})}\n\n"
                    web_sources = search_web_for_artifact(effective_question)
                    yield f"data: {json.dumps({'type': 'web_search_complete', 'sources': web_sources})}\n\n"

        web_ctx = format_sources_for_context(web_sources) if web_sources else ""

        if image_mode:
            model_question = (
                f"The user wants an image generated with this prompt:\n\n{augmented_question}\n\n"
                "Your job is to provide detailed creative direction for this image — NOT to generate it. "
                "Describe: the subject and focal point, composition and framing, art style and medium, "
                "color palette, lighting, and overall mood. Be specific and visual. "
                "Output only your creative direction, no preamble."
            )
        else:
            model_question = augmented_question

        def run_model(name, func):
            event_queue.put({'type': 'progress', 'model': name, 'status': 'running'})
            try:
                if artifact_mode:
                    q = build_artifact_model_prompt(name, augmented_question, web_ctx, artifact_type, papers,
                                                    prior_code if is_studio_followup else "")
                else:
                    q = model_question
                result, usage = func(q, model_tokens, history)
                print(f"[DEBUG] {name} response length: {len(result) if result else 0} chars", flush=True)
                with lock:
                    results[name] = result
                    usages[name] = usage
                event_queue.put({'type': 'model_done', 'model': name, 'response': result})
            except Exception as e:
                print(f"[DEBUG] {name} error: {type(e).__name__}: {str(e)}", flush=True)
                with lock:
                    results[name] = None
                    usages[name] = None
                event_queue.put({'type': 'model_error', 'model': name, 'error': str(e)})

        model_map = MODELS_INTENSIVE if is_intensive else MODELS_GENERAL

        def _mk(fn, m): return lambda q, t, h, _m=m, _fn=fn: _fn(q, t, h, model=_m)
        all_models = [
            ('GPT',    _mk(ask_gpt,    model_map["gpt"]),    'gpt'),
            ('Claude', _mk(ask_claude, model_map["claude"]), 'claude'),
            ('Gemini', _mk(ask_gemini, model_map["gemini"]), 'gemini'),
            ('Grok',   _mk(ask_grok,   MODELS_GENERAL["grok"]), 'grok'),
        ]
        # In image mode, Gemini generates images only — exclude it from text queries
        text_keys = requested_models - {'gemini'} if image_mode else requested_models
        models = [(name, func) for name, func, key in all_models if key in text_keys]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(len(models), 1))
        for name, func in models:
            executor.submit(run_model, name, func)

        completed = 0
        while completed < len(models):
            try:
                event = event_queue.get(timeout=120)
                if event['type'] in ('model_done', 'model_error'):
                    completed += 1
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                break

        executor.shutdown(wait=False)

        model_aliases = {'GPT': 'CGPT', 'Claude': 'CLDE', 'Gemini': 'GMNI', 'Grok': 'GRK'}
        active_responses = '\n\n'.join(
            f"{model_aliases[name]} response: {truncate(results.get(name))}"
            for name, _, key in all_models if key in text_keys
        )
        if image_mode:
            synthesis_system = SYNTHESIS_SYSTEM_IMAGE
            synthesis_prompt = f"Original user request: {augmented_question}\n\n{active_responses}"
        elif artifact_mode:
            synthesis_system = (
                "You are an expert engineer writing complete, working code."
                if artifact_type == "code" else
                "You are a research editor producing a structured, publishable artifact. "
                "Follow the output format in the prompt exactly. Every section must earn its place — "
                "include something only if understanding it changes how you think about the topic. "
                "Show genuine consensus AND genuine disagreement across perspectives. "
                "Write as a single expert author. Use inline citations [1], [2] throughout."
            )
            synthesis_prompt = get_artifact_synthesis_prompt(
                augmented_question,
                {n: r for n, r in results.items() if r},
                web_sources,
                artifact_type,
                papers,
                prior_code if is_studio_followup else "",
            )
        else:
            synthesis_system = SYNTHESIS_SYSTEM_TEXT
            synthesis_prompt = f"Original question: {augmented_question}\n\n{active_responses}"

        yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"

        synth_model_id = {k: v for k, v in model_map.items()}
        synth_model_id.setdefault('grok', MODELS_GENERAL['grok'])
        council_text = ''
        actual_synth = 'perplexity' if do_perplexity else synthesizer
        try:
            if do_perplexity:
                n = sum(1 for _, _, key in all_models if key in text_keys)
                web_synthesis_prompt = (
                    f"You are synthesizing responses from {n} AI models on the following question. "
                    f"You also have access to live web search — use it to verify claims, add current "
                    f"information, and ground the synthesis in real sources.\n\n"
                    f"Original question: {augmented_question}\n\n"
                    f"{active_responses}\n\n"
                    f"Your task:\n"
                    f"1. Identify where the models agree and where they diverge\n"
                    f"2. Use your web search access to verify key claims and add any current information the models may have missed\n"
                    f"3. Synthesize the most accurate, complete, and up to date answer\n"
                    f"4. Cite your web sources inline using [1], [2] etc\n"
                    f"5. Note if any model responses contained outdated or incorrect information based on what you found\n\n"
                    f"Provide a synthesized response that is grounded in both the models' reasoning and current web sources."
                )
                council_text, citations = synthesize_with_perplexity(web_synthesis_prompt, synthesis_tokens, search_query=req.question)
                yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': council_text})}\n\n"
                yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
                if citations:
                    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            else:
                synth_map = {
                    'claude': synthesize_with_claude,
                    'gpt': synthesize_with_gpt,
                    'gemini': synthesize_with_gemini,
                    'grok': synthesize_with_grok,
                }
                synth_fn = synth_map.get(actual_synth, synthesize_with_claude)
                synth_model_str = model_map.get(actual_synth) or MODELS_GENERAL.get(actual_synth, MODELS_GENERAL["claude"])
                synth_history = history if (is_studio_followup and prior_code) else ([] if artifact_mode else history)
                for text_chunk in synth_fn(synthesis_prompt, synthesis_tokens, synth_history, system=synthesis_system, model=synth_model_str):
                    council_text += text_chunk
                    yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': text_chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
                if artifact_mode:
                    if papers:
                        # Verify citations are within the paper list and flag unsupported claims
                        is_valid, errors = verify_artifact_citations(council_text, papers)
                        if is_valid:
                            yield f"data: {json.dumps({'type': 'citation_verified', 'valid': True})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'citation_warning', 'valid': False, 'errors': errors[:8]})}\n\n"
                        yield f"data: {json.dumps({'type': 'artifact_papers', 'papers': papers})}\n\n"
                    elif web_sources:
                        yield f"data: {json.dumps({'type': 'artifact_sources', 'sources': web_sources})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'council_error', 'error': str(e)})}\n\n"

        if image_mode:
            yield f"data: {json.dumps({'type': 'imagen_start', 'model': 'Gemini'})}\n\n"
            try:
                enriched = enrich_prompt_for_imagen(question, results)
                images = call_gemini_imagen(enriched)
                yield f"data: {json.dumps({'type': 'images_ready', 'model': 'Gemini', 'images': images, 'enriched_prompt': enriched})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'imagen_error', 'model': 'Gemini', 'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        # ── Write query log ───────────────────────────────────────────────────
        try:
            synth_in  = max(1, (len(synthesis_prompt) + len(synthesis_system)) // 4)
            synth_out = max(1, len(council_text) // 4)
            synth_mid = synth_model_id.get(actual_synth, 'claude-sonnet-4-6')
            synth_cost = _cost(synth_mid, synth_in, synth_out)

            model_log = {}
            total_cost = synth_cost
            for name, u in usages.items():
                if u:
                    c = _cost(u["model"], u["input_tokens"], u["output_tokens"])
                    total_cost += c
                    model_log[name] = {**u, "cost_usd": round(c, 8)}

            _write_log({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_preview": effective_question[:120],
                "models_used": list(requested_models),
                "synthesizer": actual_synth,
                "image_mode": image_mode,
                "attachments": len(attachments),
                "models": model_log,
                "synthesizer_usage": {
                    "model": synth_mid,
                    "input_tokens": synth_in,
                    "output_tokens": synth_out,
                    "cost_usd": round(synth_cost, 8),
                    "note": "token counts estimated from text length",
                },
                "total_cost_usd": round(total_cost, 8),
            })
        except Exception as log_err:
            print(f"[LOG] Failed to write query log: {log_err}", flush=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class ReviewRequest(BaseModel):
    code: str
    language: str = "auto"
    context: str = ""
    filename: str = ""

@app.post("/review")
def code_review(req: ReviewRequest, request: Request):
    code = req.code.strip()
    if not code:
        def err():
            yield 'data: {"type":"error","message":"No code provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    allowed, count = check_and_increment_ip_limit(request)
    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":DAILY_LIMIT})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    language_hint = f"Language: {req.language}" if req.language != "auto" else ""
    filename_hint = f"Filename: {req.filename}" if req.filename else ""
    context_hint = f"Context: {req.context}" if req.context else ""
    meta = "\n".join(filter(None, [language_hint, filename_hint, context_hint]))

    security_prompt = f"""You are a senior security engineer conducting a security audit.

{meta}

Code to audit:
{code}

YOUR MANDATE: Find ONLY security issues. Do not comment on code quality or performance.

Look specifically for:
- Injection vulnerabilities (SQL, command, XSS, SSTI)
- Authentication and authorization flaws
- Hardcoded secrets, API keys, passwords, tokens
- Insecure cryptography or hashing
- Path traversal vulnerabilities
- Insecure deserialization
- Missing input validation or sanitization
- Exposed sensitive data or PII
- Insecure dependencies or imports
- CORS misconfigurations
- Race conditions with security implications

Return ONLY a JSON array. No preamble, no markdown:
[
  {{
    "severity": "critical|high|medium|low",
    "title": "Short issue title",
    "file": "filename if the code came from a named file, else omit",
    "line": 47,
    "description": "Clear explanation of the vulnerability",
    "impact": "What an attacker could do with this",
    "fix": "Specific code or approach to fix it"
  }}
]

If no security issues found return an empty array: []"""

    quality_prompt = f"""You are a senior software engineer conducting a code quality review.

{meta}

Code to review:
{code}

YOUR MANDATE: Find ONLY code quality issues. Do not comment on security or performance.

Look specifically for:
- Logic errors and bugs
- Edge cases that are not handled
- Null pointer or undefined access risks
- Incorrect error handling or swallowed exceptions
- Dead code or unreachable branches
- Misleading variable or function names
- Functions doing too many things
- Missing or incorrect type handling
- Off-by-one errors
- Incorrect assumptions about data
- Code that works but will break under certain inputs

Return ONLY a JSON array. No preamble, no markdown:
[
  {{
    "severity": "critical|high|medium|low",
    "title": "Short issue title",
    "file": "filename if the code came from a named file, else omit",
    "line": 23,
    "description": "Clear explanation of the issue",
    "impact": "What goes wrong when this triggers",
    "fix": "Specific code or approach to fix it"
  }}
]

If no quality issues found return an empty array: []"""

    performance_prompt = f"""You are a senior performance engineer conducting a performance audit.

{meta}

Code to audit:
{code}

YOUR MANDATE: Find ONLY performance issues. Do not comment on security or code quality.

Look specifically for:
- N+1 query problems
- Missing database indexes or inefficient queries
- Unnecessary loops or nested iterations
- Memory leaks or excessive memory allocation
- Blocking synchronous operations that should be async
- Missing caching for expensive operations
- Inefficient data structures for the use case
- Repeated expensive computations that could be memoized
- Large data loaded into memory when streaming would work
- Unnecessary network calls or API requests in loops

Return ONLY a JSON array. No preamble, no markdown:
[
  {{
    "severity": "critical|high|medium|low",
    "title": "Short issue title",
    "file": "filename if the code came from a named file, else omit",
    "line": 89,
    "description": "Clear explanation of the performance issue",
    "impact": "How this affects speed, memory, or cost at scale",
    "fix": "Specific code or approach to fix it"
  }}
]

If no performance issues found return an empty array: []"""

    def generate():
        results = {"security": [], "quality": [], "performance": []}
        lock = threading.Lock()
        event_queue = queue.Queue()

        def run_agent(agent_name, model_func, prompt):
            event_queue.put({'type': 'agent_start', 'agent': agent_name})
            try:
                import re as _re
                response, usage = model_func(prompt, 2000)
                clean = _re.sub(r'```(?:json)?|```', '', response).strip()
                try:
                    issues = json.loads(clean)
                    if not isinstance(issues, list):
                        issues = []
                except (json.JSONDecodeError, ValueError):
                    issues = []
                with lock:
                    results[agent_name] = issues
                event_queue.put({'type': 'agent_done', 'agent': agent_name, 'issues': issues, 'count': len(issues)})
            except Exception as e:
                event_queue.put({'type': 'agent_error', 'agent': agent_name, 'error': str(e)})

        agents = [
            ('security',    ask_gpt,    security_prompt),
            ('quality',     ask_claude, quality_prompt),
            ('performance', ask_gemini, performance_prompt),
        ]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        for name, func, prompt in agents:
            executor.submit(run_agent, name, func, prompt)

        completed = 0
        while completed < 3:
            try:
                event = event_queue.get(timeout=120)
                if event['type'] in ('agent_done', 'agent_error'):
                    completed += 1
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                break

        executor.shutdown(wait=False)

        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_issues = []
        for agent_name, issues in results.items():
            for issue in issues:
                all_issues.append({**issue, 'agent': agent_name})
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
        yield f"data: {json.dumps({'type': 'review_complete', 'issues': all_issues, 'counts': {k: len(v) for k, v in results.items()}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── Research Explorer ─────────────────────────────────────────────────────────
# Three models each take a distinct research angle (historical, conceptual,
# applied), a fourth call drafts a Mermaid diagram, a fifth proposes follow-up
# questions, and Claude synthesizes the three perspectives into one exploration.
@app.post("/research")
def research(req: ResearchRequest, request: Request):
    topic = req.topic.strip()
    if not topic:
        def err():
            yield 'data: {"type":"error","message":"No topic provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # Research is a heavy multi-model studio operation — bill it against the
    # existing intensive bucket (same limit Studio Build/Research used before).
    allowed, count = check_and_increment_intensive_limit(request)
    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":INTENSIVE_LIMIT,"mode":"intensive"})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    follow_up_context = (
        f"\nThis is a follow-up exploration branching from the topic: {req.follow_up_of}"
        if req.follow_up_of else ""
    )
    extra_context = f"\n\nAdditional context to consider: {req.context}" if req.context.strip() else ""

    gpt_prompt = f"""You are a research analyst specializing in historical context and factual grounding.

Topic: {topic}{follow_up_context}{extra_context}

YOUR MANDATE: Research this topic from a historical and factual angle. Cover:
- How this topic developed over time
- Key events, milestones, or turning points
- Established facts and data points
- Real world examples and case studies
- What the evidence actually shows

Be specific. Cite real examples. Avoid vague generalities.
Write 3-4 substantive paragraphs. Do not use bullet points."""

    claude_prompt = f"""You are a conceptual analyst specializing in deep theoretical understanding.

Topic: {topic}{follow_up_context}{extra_context}

YOUR MANDATE: Research this topic from a conceptual and theoretical angle. Cover:
- The underlying principles and frameworks
- Why things work the way they do at a fundamental level
- Competing theories or schools of thought
- What experts debate and why
- The most important ideas someone needs to understand this topic deeply

Be intellectually rigorous. Surface non-obvious insights.
Write 3-4 substantive paragraphs. Do not use bullet points."""

    gemini_prompt = f"""You are an applied research specialist focusing on practical implications and future directions.

Topic: {topic}{follow_up_context}{extra_context}

YOUR MANDATE: Research this topic from a practical and forward-looking angle. Cover:
- How this topic applies in the real world today
- Current challenges and open problems
- Emerging trends and future directions
- Who is working on this and what they are building
- What someone could actually do with this knowledge

Be concrete and actionable. Focus on what is happening now and next.
Write 3-4 substantive paragraphs. Do not use bullet points."""

    diagram_prompt = f"""You are a knowledge visualization expert.

Topic: {topic}

Analyze this topic and decide which diagram type best represents it:
- "flowchart" — if the topic involves a process, system, or causal chain
- "mindmap" — if the topic has many interconnected concepts branching from a center
- "timeline" — if the topic has important historical progression or stages
- "quadrantChart" — if the topic involves comparing things across two dimensions
- "graph" — if the topic involves a network of relationships

Then generate valid Mermaid.js diagram syntax for that diagram type.

Rules:
- Return ONLY a JSON object, no markdown, no backticks
- The diagram must be valid Mermaid syntax that renders without errors
- For flowchart use "flowchart TD" and arrows like A[Label] --> B[Label]
- For graph use "graph TD" with the same arrow syntax
- Keep it focused — 6 to 12 nodes maximum
- Use clear, short labels on each node
- Do not use parentheses, quotes, or special characters inside node labels

Format:
{{
  "diagram_type": "mindmap",
  "title": "Short descriptive title for the diagram",
  "mermaid": "mindmap\\n  root((Urban Development))\\n    Housing\\n      Affordable\\n      Zoning\\n    Transport\\n      Transit\\n      Roads\\n    Economy\\n      Jobs\\n      Investment"
}}"""

    questions_prompt = f"""You are an intellectual guide helping someone explore a topic deeply.

Topic: {topic}

Generate exactly 5 follow-up questions that would genuinely deepen someone's understanding of this topic.

Rules:
- Questions should be intellectually interesting, not obvious
- Each question should open a genuinely different angle of exploration
- Questions should be specific enough to be researchable
- Avoid yes/no questions — each should invite deep exploration
- Think about what a curious expert would want to explore next

Return ONLY a JSON array of 5 strings. No preamble, no markdown:
["Question one?", "Question two?", "Question three?", "Question four?", "Question five?"]"""

    def generate():
        import re as _re
        results = {}
        lock = threading.Lock()
        event_queue = queue.Queue()

        def run_model(name, func, prompt, max_tokens=2000):
            event_queue.put({'type': 'model_start', 'model': name})
            try:
                response, _usage = func(prompt, max_tokens)
                with lock:
                    results[name] = response or ""
                event_queue.put({'type': 'model_done', 'model': name, 'response': response or ""})
            except Exception as e:
                with lock:
                    results[name] = ""
                event_queue.put({'type': 'model_error', 'model': name, 'error': str(e)})

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for name, func, prompt in [
            ('GPT',    ask_gpt,    gpt_prompt),
            ('Claude', ask_claude, claude_prompt),
            ('Gemini', ask_gemini, gemini_prompt),
        ]:
            executor.submit(run_model, name, func, prompt)
        executor.submit(run_model, 'diagram',   ask_claude, diagram_prompt, 1200)
        executor.submit(run_model, 'questions', ask_gpt,    questions_prompt, 800)

        completed = 0
        while completed < 5:
            try:
                event = event_queue.get(timeout=120)
            except queue.Empty:
                break

            etype = event['type']
            model = event.get('model')

            if etype == 'model_start':
                # Surface loading state only for the three visible research models
                if model in ('GPT', 'Claude', 'Gemini'):
                    yield f"data: {json.dumps(event)}\n\n"
                continue

            # model_done or model_error — both count toward completion
            completed += 1

            if etype == 'model_error':
                if model in ('GPT', 'Claude', 'Gemini'):
                    yield f"data: {json.dumps(event)}\n\n"
                elif model == 'diagram':
                    yield f"data: {json.dumps({'type': 'diagram_error', 'error': event.get('error', '')})}\n\n"
                elif model == 'questions':
                    yield f"data: {json.dumps({'type': 'questions_error', 'error': event.get('error', '')})}\n\n"
                continue

            # model_done
            if model in ('GPT', 'Claude', 'Gemini'):
                yield f"data: {json.dumps(event)}\n\n"
            elif model == 'diagram':
                try:
                    clean = _re.sub(r'```(?:json)?|```', '', event['response']).strip()
                    diagram_data = json.loads(clean)
                    yield f"data: {json.dumps({'type': 'diagram_ready', 'diagram': diagram_data})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'diagram_error', 'error': str(e)})}\n\n"
            elif model == 'questions':
                try:
                    clean = _re.sub(r'```(?:json)?|```', '', event['response']).strip()
                    questions = json.loads(clean)
                    if not isinstance(questions, list):
                        questions = []
                    yield f"data: {json.dumps({'type': 'questions_ready', 'questions': questions})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'questions_error', 'error': str(e)})}\n\n"

        executor.shutdown(wait=False)

        synthesis_prompt = f"""You are synthesizing three research perspectives on: {topic}

Historical/Factual perspective:
{results.get('GPT', 'Not available')}

Conceptual/Theoretical perspective:
{results.get('Claude', 'Not available')}

Practical/Applied perspective:
{results.get('Gemini', 'Not available')}

Synthesize these three perspectives into one cohesive, well-structured exploration.

Rules:
- Do NOT mention which model said what
- Weave all three perspectives together naturally
- Use clear section headers (## Header) to organize the content
- Start with the most important insight, not background
- Be intellectually honest about what is debated vs settled
- Total length: 600 to 900 words
- Write for a curious, intelligent reader who wants real depth"""

        yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"
        try:
            for chunk in synthesize_with_claude(synthesis_prompt, 3000):
                yield f"data: {json.dumps({'type': 'synthesis_chunk', 'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'synthesis_done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'synthesis_error', 'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# Vercel entry point — wraps the ASGI app so Vercel recognises this as a
# valid Python Serverless Function.
handler = Mangum(app)
