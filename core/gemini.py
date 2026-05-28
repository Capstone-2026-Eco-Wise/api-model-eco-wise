import google.generativeai as genai
from config.settings import GEMINI_API_KEY
from utils.logger import logger

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_tips(label: str) -> str:
    """Meminta tips dari Gemini API berdasarkan prediksi sampah."""
    # Jika objek bukan sampah, langsung return pesan bypass
    if label == "Non-Waste":
        return "Objek ini bukan merupakan kategori sampah. Tidak ada tips daur ulang yang tersedia."

    prompt = f"Berikan 2 langkah praktis dan singkat untuk mengelola atau mendaur ulang sampah berjenis {label}. Balas dalam bahasa Indonesia yang ramah."
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Tips daur ulang tidak dapat dimuat saat ini. Pastikan membuang sampah pada tempat yang sesuai."
