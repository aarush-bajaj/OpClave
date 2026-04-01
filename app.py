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
import requests

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

app = Flask(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
sarvam_api_key = os.getenv("SARVAM_API_KEY")


def ask_gpt(question):
    response = openai_client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        max_completion_tokens=100000,
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


def ask_claude(question):
    import time
    for attempt in range(5):
        try:
            with anthropic_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                messages=[{"role": "user", "content": question}]
            ) as stream:
                return stream.get_final_text()
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 4:
                time.sleep(10 * (attempt + 1))
            else:
                raise


def ask_gemini(question):
    response = gemini_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config={"maxOutputTokens": 100000},
        contents=question
    )
    return response.text


def ask_grok(question):
    response = grok_client.chat.completions.create(
        model="grok-4.20-0309-non-reasoning",
        max_tokens=100000,
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


def ask_sarvam(question):
    response = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {sarvam_api_key}", "Content-Type": "application/json"},
        json={"model": "sarvam-105b", "messages": [{"role": "user", "content": question}], "max_tokens": 100000}
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or str(message)


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
        event_queue = queue.Queue()
        results = {}
        lock = threading.Lock()

        def run_model(name, func):
            event_queue.put({'type': 'progress', 'model': name, 'status': 'running'})
            try:
                result = func(question)
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
            ('Sarvam', ask_sarvam),
        ]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for name, func in models:
            executor.submit(run_model, name, func)

        completed = 0
        while completed < len(models):
            try:
                event = event_queue.get(timeout=180)
                if event['type'] in ('model_done', 'model_error'):
                    completed += 1
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                break

        executor.shutdown(wait=False)

        synthesis_prompt = f"""You are synthesizing responses from five different AI models.

Original question: {question}

Jackfruit response: {results.get('GPT', 'No response')}
Passionfruit response: {results.get('Claude', 'No response')}
Grapefruit response: {results.get('Gemini', 'No response')}
Strawberry response: {results.get('Grok', 'No response')}
Apple response: {results.get('Sarvam', 'No response')}

Your task:
1. Identify where they agree (high confidence)
2. Identify where they disagree (uncertainty)
3. Synthesize the most accurate, complete answer
4. Note any contradictions or limitations

Provide a final synthesized response."""

        yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"

        try:
            import time
            council_response = None
            for attempt in range(3):
                try:
                    with anthropic_client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=16000,
                        messages=[{"role": "user", "content": synthesis_prompt}]
                    ) as stream_obj:
                        council_response = stream_obj.get_final_text()
                    break
                except Exception as e:
                    if "overloaded" in str(e).lower() and attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise
            yield f"data: {json.dumps({'type': 'council_done', 'response': council_response})}\n\n"
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
