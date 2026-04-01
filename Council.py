from openai import OpenAI
from anthropic import Anthropic
from google import genai
from dotenv import load_dotenv
import os
import sys
import time
import threading
import concurrent.futures
import requests

load_dotenv(dotenv_path="/Users/aarushbajaj/Desktop/Council/.env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
grok_client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
sarvam_api_key = os.getenv("SARVAM_API_KEY")

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Progress:
    """Multi-line animated progress display."""

    def __init__(self, tasks):
        self.tasks = tasks
        self.status = {t: "pending" for t in tasks}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lines_printed = 0

    def mark_running(self, task):
        with self.lock:
            self.status[task] = "running"

    def mark_done(self, task):
        with self.lock:
            self.status[task] = "done"

    def mark_error(self, task):
        with self.lock:
            self.status[task] = "error"

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._render(final=True)

    def _render(self, spin_idx=0, final=False):
        lines = []
        with self.lock:
            for t in self.tasks:
                s = self.status[t]
                if s == "running":
                    icon = SPINNER[spin_idx % len(SPINNER)] if not final else "✓"
                elif s == "done":
                    icon = "✓"
                elif s == "error":
                    icon = "✗"
                else:
                    icon = "·"
                lines.append(f"  {icon} {t}")

        if self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A")
        for line in lines:
            sys.stdout.write(f"\r\033[2K{line}\n")
        sys.stdout.flush()
        self._lines_printed = len(lines)

    def _run(self):
        i = 0
        while not self._stop.is_set():
            self._render(spin_idx=i)
            i += 1
            time.sleep(0.1)


def spin_label(label, stop_event):
    """Single-line spinner for a labeled task."""
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{SPINNER[i % len(SPINNER)]} {label}...")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write(f"\r✓ {label}    \n")
    sys.stdout.flush()


def is_overloaded(e):
    return getattr(e, "status_code", None) == 529 or "overloaded" in str(e).lower()


def ask_gpt(question, progress):
    progress.mark_running("GPT")
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-5.2-2025-12-11",
                max_completion_tokens=100000,
                messages=[{"role": "user", "content": question}]
            )
            progress.mark_done("GPT")
            return response.choices[0].message.content, response.usage.completion_tokens
        except Exception as e:
            if is_overloaded(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                progress.mark_error("GPT")
                raise


def ask_claude(question, progress):
    progress.mark_running("Claude")
    for attempt in range(3):
        try:
            with anthropic_client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=100000,
                messages=[{"role": "user", "content": question}]
            ) as stream:
                final_message = stream.get_final_message()
            progress.mark_done("Claude")
            text = "".join(block.text for block in final_message.content if hasattr(block, "text"))
            return text, final_message.usage.output_tokens
        except Exception as e:
            if is_overloaded(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                progress.mark_error("Claude")
                raise


def ask_gemini(question, progress):
    progress.mark_running("Gemini")
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-pro-preview",
                config={"maxOutputTokens": 100000},
                contents=question
            )
            progress.mark_done("Gemini")
            tokens = response.usage_metadata.candidates_token_count
            return response.text, tokens
        except Exception as e:
            if is_overloaded(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                progress.mark_error("Gemini")
                raise


def ask_grok(question, progress):
    progress.mark_running("Grok")
    for attempt in range(3):
        try:
            response = grok_client.chat.completions.create(
                model="grok-4.20-0309-non-reasoning",
                max_tokens=100000,
                messages=[{"role": "user", "content": question}]
            )
            progress.mark_done("Grok")
            return response.choices[0].message.content, response.usage.completion_tokens
        except Exception as e:
            if is_overloaded(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                progress.mark_error("Grok")
                raise


def ask_sarvam(question, progress):
    progress.mark_running("Sarvam")
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.sarvam.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {sarvam_api_key}", "Content-Type": "application/json"},
                json={"model": "sarvam-105b", "messages": [{"role": "user", "content": question}], "max_tokens": 100000}
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            progress.mark_done("Sarvam")
            return message.get("content") or message.get("reasoning_content") or str(message), tokens
        except Exception as e:
            if is_overloaded(e) and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                progress.mark_error("Sarvam")
                raise


user_question = input("Enter your question: ").strip()

print("\nQuerying models in parallel:")
progress = Progress(["GPT", "Claude", "Gemini", "Grok", "Sarvam"]).start()

with concurrent.futures.ThreadPoolExecutor() as executor:
    gpt_future = executor.submit(ask_gpt, user_question, progress)
    claude_future = executor.submit(ask_claude, user_question, progress)
    gemini_future = executor.submit(ask_gemini, user_question, progress)
    grok_future = executor.submit(ask_grok, user_question, progress)
    sarvam_future = executor.submit(ask_sarvam, user_question, progress)

    gpt_response, gpt_tokens = gpt_future.result()
    claude_response, claude_tokens = claude_future.result()
    gemini_response, gemini_tokens = gemini_future.result()
    grok_response, grok_tokens = grok_future.result()
    sarvam_response, sarvam_tokens = sarvam_future.result()

progress.stop()

print("\nOutput tokens used:")
print(f"  GPT:    {gpt_tokens:,}")
print(f"  Claude: {claude_tokens:,}")
print(f"  Gemini: {gemini_tokens:,}")
print(f"  Grok:   {grok_tokens:,}")
print(f"  Sarvam: {sarvam_tokens:,}")

with open("gpt_response.txt", "w", encoding="utf-8") as f:
    f.write(gpt_response)
with open("claude_response.txt", "w", encoding="utf-8") as f:
    f.write(claude_response)
with open("gemini_response.txt", "w", encoding="utf-8") as f:
    f.write(gemini_response)
with open("grok_response.txt", "w", encoding="utf-8") as f:
    f.write(grok_response)
with open("sarvam_response.txt", "w", encoding="utf-8") as f:
    f.write(sarvam_response)

synthesis_prompt = f"""You are synthesizing responses from five different people.

Original question: {user_question}

Jackfruit response: {gpt_response}
Passionfruit response: {claude_response}
Grapefruit response: {gemini_response}
Strawberry response: {grok_response}
Apple response: {sarvam_response}

Your task:
1. Identify where they agree (high confidence)
2. Identify where they disagree (uncertainty)
3. Synthesize the most accurate, complete answer
4. Note any contradictions or limitations

Provide a final synthesized response."""

print()
stop_synth = threading.Event()
synth_thread = threading.Thread(
    target=spin_label, args=("Synthesizing final answer", stop_synth), daemon=True
)
synth_thread.start()

for attempt in range(3):
    try:
        with anthropic_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            messages=[{"role": "user", "content": synthesis_prompt}]
        ) as stream:
            synth_message = stream.get_final_message()
        break
    except Exception as e:
        if is_overloaded(e) and attempt < 2:
            time.sleep(2 ** attempt)
        else:
            stop_synth.set()
            synth_thread.join()
            raise

stop_synth.set()
synth_thread.join()

final_text = "".join(block.text for block in synth_message.content if hasattr(block, "text"))
synth_tokens = synth_message.usage.output_tokens

print(f"  Synthesis output tokens: {synth_tokens:,}")

with open("council_final_answer.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("\nResponses saved:")
print("  • gpt_response.txt")
print("  • claude_response.txt")
print("  • gemini_response.txt")
print("  • grok_response.txt")
print("  • sarvam_response.txt")
print("  • council_final_answer.txt")
