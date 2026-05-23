from google import genai

from utils.env import env_vars

client = genai.Client(api_key=env_vars.get("GEMINI_API_KEY"))

print("List Model yang Tersedia dari API KEY:")
print("-" * 50)
for model in client.models.list():
    # Hanya tampilkan model yang mendukung generateContent (teks)
    if "generateContent" in model.supported_actions:
        print(model.name)
print("-" * 50)
