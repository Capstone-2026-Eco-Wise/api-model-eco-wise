from google import genai

# Masukkan API Key aslimu di bawah ini
API_KEY = "MASUKKAN API KEY ASLIMU DISINI"

client = genai.Client(api_key=API_KEY)

print("Daftar model yang tersedia untuk API Key kamu:")
print("-" * 50)
for model in client.models.list():
    # Hanya tampilkan model yang mendukung generateContent (teks)
    if "generateContent" in model.supported_actions:
        print(model.name)
print("-" * 50)
