from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/Users/aarushbajaj/Desktop/Council/.env")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

user_question = input("Ask something: ")

response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # cheapest claude model
    max_tokens=8000,
    messages=[
        {"role": "user", "content": user_question}
    ]
)

print(response.content[0].text)