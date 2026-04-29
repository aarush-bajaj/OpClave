from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from mangum import Mangum
from pydantic import BaseModel
from openai import OpenAI
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
    "gemini": "gemini-3.1-flash-lite-preview",
    "grok":   "grok-4-1-fast-non-reasoning",
}
MODELS_INTENSIVE = {
    "gpt":    "gpt-5.5-2026-04-23",
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


def check_and_increment_ip_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, current_count) for the given IP. Resets every 24h."""
    try:
        key = f"council:prompts:{ip}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, 86400])
        return (count <= DAILY_LIMIT, int(count))
    except Exception:
        return (True, 0)  # fail open so an Upstash outage doesn't break the site


def check_and_increment_intensive_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, current_count) for intensive queries. Separate 3/day bucket."""
    try:
        key = f"council:intensive:{ip}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, 86400])
        return (count <= INTENSIVE_LIMIT, int(count))
    except Exception:
        return (True, 0)


@app.get("/assets/logo.png")
def serve_logo():
    return FileResponse(os.path.join(_root, "templates", "assets", "logo.png"), media_type="image/png")

@app.get("/assets/BrandLogo.png")
def serve_brand_logo():
    return FileResponse(os.path.join(_root, "templates", "assets", "BrandLogo.png"), media_type="image/png")

@app.get("/assets/tailwind.css")
def serve_tailwind():
    return FileResponse(os.path.join(_root, "templates", "assets", "tailwind.css"), media_type="text/css")

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
    response = openai_client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    usage = {"model": model,
             "input_tokens": response.usage.prompt_tokens,
             "output_tokens": response.usage.completion_tokens}
    return response.choices[0].message.content, usage


def ask_claude(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["claude"]
    messages = list(history or []) + [{"role": "user", "content": question}]
    for attempt in range(3):
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
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def ask_gemini(question, max_tokens, history=None, model=None):
    model = model or MODELS_GENERAL["gemini"]
    contents = []
    for msg in (history or []):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
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
    for attempt in range(3):
        try:
            with anthropic_client.messages.stream(**kwargs) as stream_obj:
                for chunk in stream_obj.text_stream:
                    yield chunk
            return
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def synthesize_with_gpt(prompt, max_tokens, history=None, system=None, model=None):
    model = model or MODELS_GENERAL["gpt"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages += list(history or []) + [{"role": "user", "content": prompt}]
    stream = openai_client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=messages,
        stream=True,
    )
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
    for chunk in gemini_client.models.generate_content_stream(
        model=model,
        config=config,
        contents=contents,
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

class FeedbackRequest(BaseModel):
    message: str = ""

class AdminVerifyRequest(BaseModel):
    code: str = ""


# ── Perplexity Sonar (web-aware council member) ───────────────────────────────

def synthesize_with_perplexity(prompt: str, max_tokens: int):
    response = perplexity_client.chat.completions.create(
        model="sonar",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    with open(os.path.join(_root, "templates", "landing.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/app")
def app_route():
    with open(os.path.join(_root, "templates", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


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

    host = request.headers.get("host", "")
    is_local = host.startswith("localhost") or host.startswith("127.")
    if is_local:
        print(f"[feedback] {message}")
        return {"ok": True}

    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_pass:
        return {"ok": False, "error": "Email not configured"}

    def send_email():
        import smtplib, email.mime.text, email.mime.multipart
        sender = "aarushbajajus@gmail.com"
        recipient = "councilaiclb@gmail.com"
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = "Opclave — Feedback"
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

    # Enforce intensive-mode restrictions before anything else
    if is_intensive:
        requested_models -= {"grok"}
        if synthesizer == "grok":
            synthesizer = "claude"

    if not question and not attachments:
        def err():
            yield 'data: {"type":"error","message":"No question provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

    if is_intensive:
        allowed, count = check_and_increment_intensive_limit(ip)
        active_limit = INTENSIVE_LIMIT
    else:
        allowed, count = check_and_increment_ip_limit(ip)
        active_limit = DAILY_LIMIT

    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":active_limit,"mode":req.mode})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    attachment_ctx = build_attachment_context(attachments)
    effective_question = question or "Please review the attached file(s)."
    augmented_question = f"{attachment_ctx}\n{effective_question}" if attachment_ctx else effective_question
    do_perplexity = req.web_search and not image_mode and not is_intensive

    def generate():
        nonlocal augmented_question
        model_tokens = estimate_max_tokens(effective_question)
        synthesis_tokens = min(max(model_tokens * 2, 3000), 6000)
        print(f"[DEBUG] Question: {repr(effective_question[:80])}", flush=True)
        print(f"[DEBUG] model_tokens={model_tokens}  synthesis_tokens={synthesis_tokens}", flush=True)
        print(f"[DEBUG] Active models: {requested_models}  Synthesizer: {synthesizer}", flush=True)

        event_queue = queue.Queue()
        results = {}
        usages = {}
        lock = threading.Lock()

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
                result, usage = func(model_question, model_tokens, history)
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
                council_text, citations = synthesize_with_perplexity(web_synthesis_prompt, synthesis_tokens)
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
                for text_chunk in synth_fn(synthesis_prompt, synthesis_tokens, history, system=synthesis_system, model=synth_model_str):
                    council_text += text_chunk
                    yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': text_chunk})}\n\n"
                yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
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


# Vercel entry point — wraps the ASGI app so Vercel recognises this as a
# valid Python Serverless Function.
handler = Mangum(app)
