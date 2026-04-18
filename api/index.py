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
import urllib.request
import urllib.parse
import hmac
import base64
from google.genai import types

from datetime import datetime, timezone
from typing import List

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_root, '.env'))

app = FastAPI()

DAILY_LIMIT = 15
_UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
_UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")


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


def check_and_increment_global_limit() -> tuple[bool, int]:
    """Returns (allowed, current_count). Increments counter; sets 24h TTL on first use."""
    try:
        key = f"council:prompts:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        count = _redis(["INCR", key])
        if count == 1:
            _redis(["EXPIRE", key, 86400])
        return (count <= DAILY_LIMIT, int(count))
    except Exception:
        return (True, 0)  # fail open so an Upstash outage doesn't break the site


@app.get("/assets/logo.png")
def serve_logo():
    return FileResponse(os.path.join(_root, "templates", "assets", "logo.png"), media_type="image/png")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")


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

def ask_gpt(question, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": question}]
    response = openai_client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content


def ask_claude(question, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": question}]
    for attempt in range(3):
        try:
            with anthropic_client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=messages,
            ) as stream:
                return stream.get_final_text()
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def ask_gemini(question, max_tokens, history=None):
    contents = []
    for msg in (history or []):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    response = gemini_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config={"maxOutputTokens": max_tokens},
        contents=contents,
    )
    return response.text


def ask_grok(question, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": question}]
    response = grok_client.chat.completions.create(
        model="grok-4.20-0309-non-reasoning",
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content


# ── Synthesis generators ──────────────────────────────────────────────────────

def synthesize_with_claude(prompt, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            with anthropic_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=messages,
            ) as stream_obj:
                for chunk in stream_obj.text_stream:
                    yield chunk
            return
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def synthesize_with_gpt(prompt, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    stream = openai_client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        max_completion_tokens=max_tokens,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def synthesize_with_gemini(prompt, max_tokens, history=None):
    contents = []
    for msg in (history or []):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    response = gemini_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config={"maxOutputTokens": max_tokens},
        contents=contents,
    )
    if response.text:
        yield response.text


def synthesize_with_grok(prompt, max_tokens, history=None):
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    stream = grok_client.chat.completions.create(
        model="grok-4.20-0309-non-reasoning",
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

class FeedbackRequest(BaseModel):
    message: str = ""

class AdminVerifyRequest(BaseModel):
    code: str = ""


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
def feedback(req: FeedbackRequest, request: Request):
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

    try:
        import smtplib, email.mime.text, email.mime.multipart
        sender = "aarushbajajus@gmail.com"
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = sender
        msg["To"] = sender
        msg["Subject"] = "Council AI — Feedback"
        msg.attach(email.mime.text.MIMEText(
            f"Feedback received:\n\n{message}\n\n---\nSent via Council AI", "plain"
        ))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, gmail_pass)
            server.sendmail(sender, sender, msg.as_string())
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/stream")
def stream(req: StreamRequest):
    question = req.question.strip()
    synthesizer = req.synthesizer.lower()
    requested_models = set(req.models)
    history = req.history
    image_mode = req.image_mode
    attachments = req.attachments

    if not question and not attachments:
        def err():
            yield 'data: {"type":"error","message":"No question provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    allowed, count = check_and_increment_global_limit()
    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":DAILY_LIMIT})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    attachment_ctx = build_attachment_context(attachments)
    effective_question = question or "Please review the attached file(s)."
    augmented_question = f"{attachment_ctx}\n{effective_question}" if attachment_ctx else effective_question

    def generate():
        model_tokens = estimate_max_tokens(effective_question)
        synthesis_tokens = min(max(model_tokens * 2, 3000), 6000)
        print(f"[DEBUG] Question: {repr(effective_question[:80])}", flush=True)
        print(f"[DEBUG] model_tokens={model_tokens}  synthesis_tokens={synthesis_tokens}", flush=True)
        print(f"[DEBUG] Active models: {requested_models}  Synthesizer: {synthesizer}", flush=True)

        event_queue = queue.Queue()
        results = {}
        lock = threading.Lock()

        def run_model(name, func):
            event_queue.put({'type': 'progress', 'model': name, 'status': 'running'})
            try:
                result = func(augmented_question, model_tokens, history)
                print(f"[DEBUG] {name} response length: {len(result) if result else 0} chars", flush=True)
                with lock:
                    results[name] = result
                event_queue.put({'type': 'model_done', 'model': name, 'response': result})
            except Exception as e:
                print(f"[DEBUG] {name} error: {type(e).__name__}: {str(e)}", flush=True)
                with lock:
                    results[name] = None
                event_queue.put({'type': 'model_error', 'model': name, 'error': str(e)})

        all_models = [
            ('GPT',    ask_gpt,    'gpt'),
            ('Claude', ask_claude, 'claude'),
            ('Gemini', ask_gemini, 'gemini'),
            ('Grok',   ask_grok,   'grok'),
        ]
        # In image mode Gemini generates images; exclude it from text queries
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
        n = len(text_keys)
        if image_mode:
            synthesis_prompt = f"""You are synthesizing creative direction from {n} AI model{'s' if n != 1 else ''} for an image generation task.

Original user request: {augmented_question}

{active_responses}

Your task:
1. Summarize the visual direction the models agreed on (style, colors, composition, mood)
2. Note any interesting differences in their creative suggestions
3. Inform the user that **4 image variants are being generated in the Gemini tab** based on their combined suggestions
4. Invite them to check the Gemini tab to browse the visuals

Keep your response concise — the visuals speak for themselves."""
        else:
            synthesis_prompt = f"""You are synthesizing responses from {n} AI model{'s' if n != 1 else ''}.

Original question: {augmented_question}

{active_responses}

Your task:
1. Identify where they agree (high confidence)
2. Identify where they disagree (uncertainty)
3. Synthesize the most accurate, complete answer
4. Note any contradictions or limitations

Provide a final synthesized response."""

        yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"

        try:
            synth_map = {
                'claude': synthesize_with_claude,
                'gpt': synthesize_with_gpt,
                'gemini': synthesize_with_gemini,
                'grok': synthesize_with_grok,
            }
            actual_synth = 'claude' if (image_mode and synthesizer == 'gemini') else synthesizer
            synth_fn = synth_map.get(actual_synth, synthesize_with_claude)
            for text_chunk in synth_fn(synthesis_prompt, synthesis_tokens, history):
                yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': text_chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'council_error', 'error': str(e)})}\n\n"

        if image_mode and 'gemini' in requested_models:
            yield f"data: {json.dumps({'type': 'imagen_start'})}\n\n"
            try:
                enriched = enrich_prompt_for_imagen(question, results)
                images = call_gemini_imagen(enriched)
                yield f"data: {json.dumps({'type': 'images_ready', 'images': images, 'enriched_prompt': enriched})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'imagen_error', 'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

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
