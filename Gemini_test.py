from google import genai
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/Users/aarushbajaj/Desktop/Council/.env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

user_question = input("Ask something: ")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=user_question
)

print(response.text)