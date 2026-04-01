import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["XAI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

def chat(messages):
    response = client.chat.completions.create(
        model="grok-4.20-0309-non-reasoning",
        messages=messages,
    )
    return response.choices[0].message.content

def main():
    messages = []
    print("Grok xAI Chat (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Grok: {reply}\n")

if __name__ == "__main__":
    main()
