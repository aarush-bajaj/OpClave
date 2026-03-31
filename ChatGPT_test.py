from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print("Key found:", os.getenv("OPENAI_API_KEY"))  # debug line

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_question = input("Ask something: ")

response = client.chat.completions.create(
    model="gpt-5-mini",
    max_completion_tokens=8000,
    messages=[
        {"role": "user", "content": user_question}
    ]
)

print(response.choices[0].message.content)