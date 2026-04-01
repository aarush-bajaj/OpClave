from dotenv import load_dotenv
import os
import requests

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

def sarvam_chat(messages: list[dict], model: str = "sarvam-105b") -> str:
    url = "https://api.sarvam.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 100000,
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or str(message)


if __name__ == "__main__":
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in .env file")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print("Sarvam-105B Chat (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = sarvam_chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Sarvam: {reply}\n")
        with open("sarvam_response.txt", "w") as f:
            f.write(reply)
