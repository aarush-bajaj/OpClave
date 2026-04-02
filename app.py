from flask import Flask, render_template, request, Response, stream_with_context
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

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

app = Flask(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")


def truncate(text, max_chars=5000):
    if not text:
        return "No response"
    return text[:max_chars] + "..." if len(text) > max_chars else text


def estimate_max_tokens(question):
    """
    Heuristic to pick an appropriate token ceiling based on question complexity.
    Tiers:
      simple   (~1 sentence factual)  → 400
      medium   (explanation/summary)  → 1200
      detailed (analysis/comparison)  → 2500
      heavy    (code/essay/long list) → 4000
    """
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
    simple_keywords = [
        'what is', 'who is', 'when did', 'where is', 'define', 'yes or no',
        'true or false', 'how many', 'how much',
    ]

    if any(kw in q for kw in heavy_keywords) or words > 60:
        return 4000
    if any(kw in q for kw in detailed_keywords) or words > 25:
        return 2500
    if any(kw in q for kw in simple_keywords) or words <= 10:
        return 400
    return 1200


def ask_gpt(question, max_tokens):
    response = openai_client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


def ask_claude(question, max_tokens):
    for attempt in range(3):
        try:
            with anthropic_client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": question}]
            ) as stream:
                return stream.get_final_text()
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def ask_gemini(question, max_tokens):
    response = gemini_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config={"maxOutputTokens": max_tokens},
        contents=question
    )
    return response.text


def ask_grok(question, max_tokens):
    response = grok_client.chat.completions.create(
        model="grok-4.20-0309-non-reasoning",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stream')
def stream():
    question = request.args.get('question', '').strip()
    if not question:
        def err():
            yield 'data: {"type":"error","message":"No question provided"}\n\n'
        return Response(err(), mimetype='text/event-stream')

    def generate():
        model_tokens = estimate_max_tokens(question)
        # Synthesis needs to cover all four responses, so give it a bit more headroom
        synthesis_tokens = min(model_tokens * 2, 4000)

        event_queue = queue.Queue()
        results = {}
        lock = threading.Lock()

        def run_model(name, func):
            event_queue.put({'type': 'progress', 'model': name, 'status': 'running'})
            try:
                result = func(question, model_tokens)
                with lock:
                    results[name] = result
                event_queue.put({'type': 'model_done', 'model': name, 'response': result})
            except Exception as e:
                with lock:
                    results[name] = None
                event_queue.put({'type': 'model_error', 'model': name, 'error': str(e)})

        models = [
            ('GPT', ask_gpt),
            ('Claude', ask_claude),
            ('Gemini', ask_gemini),
            ('Grok', ask_grok),
        ]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
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

        synthesis_prompt = f"""You are synthesizing responses from four different AI models.

Original question: {question}

Jackfruit response: {truncate(results.get('GPT'))}
Passionfruit response: {truncate(results.get('Claude'))}
Grapefruit response: {truncate(results.get('Gemini'))}
Strawberry response: {truncate(results.get('Grok'))}

Your task:
1. Identify where they agree (high confidence)
2. Identify where they disagree (uncertainty)
3. Synthesize the most accurate, complete answer
4. Note any contradictions or limitations

Provide a final synthesized response."""

        yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"

        try:
            for attempt in range(3):
                try:
                    with anthropic_client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=synthesis_tokens,
                        messages=[{"role": "user", "content": synthesis_prompt}]
                    ) as stream_obj:
                        for text_chunk in stream_obj.text_stream:
                            yield f"data: {json.dumps({'type': 'council_chunk', 'chunk': text_chunk})}\n\n"
                    break
                except Exception as e:
                    if "overloaded" in str(e).lower() and attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise
            yield f"data: {json.dumps({'type': 'council_done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'council_error', 'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
