import os
import requests

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY is not set")

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
r.raise_for_status()
models = [m["id"] for m in r.json().get("data", [])]
print("\n".join(sorted(models)))