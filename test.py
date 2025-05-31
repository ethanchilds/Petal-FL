import openai
import os

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(prompt: str, model: str = "gpt-4o", max_tokens: int = 300, temperature: float = 0.7):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response['choices'][0]['message']['content']
