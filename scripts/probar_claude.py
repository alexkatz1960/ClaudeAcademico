import os
from anthropic import Anthropic

# Leer la API Key desde el .env
from dotenv import load_dotenv
load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    messages=[{"role": "user", "content": "¿Cuál es la capital de Francia?"}]
)

print("✅ Respuesta de Claude:")
print(response.content[0].text)
