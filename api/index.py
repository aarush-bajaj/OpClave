from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from mangum import Mangum
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from dotenv import load_dotenv
import os
import json
import queue
import threading
import concurrent.futures
import time
import urllib.request
import urllib.parse
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


# ── Request model ─────────────────────────────────────────────────────────────

class StreamRequest(BaseModel):
    question: str = ""
    synthesizer: str = "claude"
    models: List[str] = ["gpt", "claude", "gemini", "grok"]
    history: List[dict] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    with open(os.path.join(_root, "templates", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/stream")
def stream(req: StreamRequest):
    question = req.question.strip()
    synthesizer = req.synthesizer.lower()
    requested_models = set(req.models)
    history = req.history

    if not question:
        def err():
            yield 'data: {"type":"error","message":"No question provided"}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    allowed, count = check_and_increment_global_limit()
    if not allowed:
        def rate_limited():
            yield f'data: {json.dumps({"type":"rate_limited","used":count,"limit":DAILY_LIMIT})}\n\n'
        return StreamingResponse(rate_limited(), media_type="text/event-stream")

    def generate():
        model_tokens = estimate_max_tokens(question)
        synthesis_tokens = min(max(model_tokens * 2, 3000), 6000)
        print(f"[DEBUG] Question: {repr(question[:80])}", flush=True)
        print(f"[DEBUG] model_tokens={model_tokens}  synthesis_tokens={synthesis_tokens}", flush=True)
        print(f"[DEBUG] Active models: {requested_models}  Synthesizer: {synthesizer}", flush=True)

        event_queue = queue.Queue()
        results = {}
        lock = threading.Lock()

        def run_model(name, func):
            event_queue.put({'type': 'progress', 'model': name, 'status': 'running'})
            try:
                result = func(question, model_tokens, history)
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
        models = [(name, func) for name, func, key in all_models if key in requested_models]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(models))
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

        model_aliases = {'GPT': 'CGPT', 'Claude': 'Clde', 'Gemini': 'GMNI', 'Grok': 'GRK'}
        active_responses = '\n\n'.join(
            f"{model_aliases[name]} response: {truncate(results.get(name))}"
            for name, _, key in all_models if key in requested_models
        )
        n = len(requested_models)
        synthesis_prompt = f"""You are synthesizing responses from {n} AI model{'s' if n != 1 else ''}.

Original question: {question}

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
            synth_fn = synth_map.get(synthesizer, synthesize_with_claude)
            for text_chunk in synth_fn(synthesis_prompt, synthesis_tokens, history):
                yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': text_chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'council_error', 'error': str(e)})}\n\n"

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
