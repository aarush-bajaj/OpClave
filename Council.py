from openai import OpenAI
from anthropic import Anthropic
from google import genai
from dotenv import load_dotenv
import os
import sys
import time
import threading
import concurrent.futures

load_dotenv(dotenv_path="/Users/aarushbajaj/Desktop/Council/.env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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


def ask_gpt(question, progress):
    progress.mark_running("GPT")
    response = openai_client.chat.completions.create(
        model="gpt-5.2-2025-12-11",
        max_completion_tokens=10000,
        messages=[{"role": "user", "content": question}]
    )
    progress.mark_done("GPT")
    return response.choices[0].message.content


def ask_claude(question, progress):
    progress.mark_running("Claude")
    with anthropic_client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=10000,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        result = stream.get_final_text()
    progress.mark_done("Claude")
    return result


def ask_gemini(question, progress):
    progress.mark_running("Gemini")
    response = gemini_client.models.generate_content(
        model="gemini-3.1-pro-preview",
        config={"maxOutputTokens": 10000},
        contents=question
    )
    progress.mark_done("Gemini")
    return response.text


user_question = input("Enter your question: ").strip()

print("\nQuerying models in parallel:")
progress = Progress(["GPT", "Claude", "Gemini"]).start()

with concurrent.futures.ThreadPoolExecutor() as executor:
    gpt_future = executor.submit(ask_gpt, user_question, progress)
    claude_future = executor.submit(ask_claude, user_question, progress)
    gemini_future = executor.submit(ask_gemini, user_question, progress)

    gpt_response = gpt_future.result()
    claude_response = claude_future.result()
    gemini_response = gemini_future.result()

progress.stop()

with open("gpt_response.txt", "w", encoding="utf-8") as f:
    f.write(gpt_response)
with open("claude_response.txt", "w", encoding="utf-8") as f:
    f.write(claude_response)
with open("gemini_response.txt", "w", encoding="utf-8") as f:
    f.write(gemini_response)

synthesis_prompt = f"""You are synthesizing responses from three different people.

Original question: {user_question} 

Jackfruit response: {gpt_response}
Passionfruit response: {claude_response}
Grapefruit response: {gemini_response}

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

with anthropic_client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=10000,
    messages=[{"role": "user", "content": synthesis_prompt}]
) as stream:
    final_text = stream.get_final_text()

stop_synth.set()
synth_thread.join()

with open("council_final_answer.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("\nResponses saved:")
print("  • gpt_response.txt")
print("  • claude_response.txt")
print("  • gemini_response.txt")
print("  • council_final_answer.txt")
#ts