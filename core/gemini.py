from google import genai
from config.settings import GEMINI_API_KEY
from utils.logger import logger

# Clean up API key (remove quotes if any)
api_key_clean = GEMINI_API_KEY.strip('"' + "'") if GEMINI_API_KEY else None

# Initialize Gemini Client if API key is provided
client = None
if api_key_clean:
    try:
        client = genai.Client(api_key=api_key_clean)
        logger.info("Gemini AI Client berhasil diinisialisasi")
    except Exception as e:
        logger.error(f"Gagal menginisialisasi Gemini Client: {e}")
else:
    logger.warning("GEMINI_API_KEY kosong atau belum diset. Fitur tips daur ulang Gemini akan dinonaktifkan.")

def get_gemini_tips(label: str) -> str:
    """Meminta tips dari Gemini API berdasarkan prediksi sampah."""
    # Jika objek bukan sampah, langsung return pesan bypass
    if label == "Non-Waste":
        return "Objek ini bukan merupakan kategori sampah. Tidak ada tips daur ulang yang tersedia."

    if not client:
        return "Tips daur ulang tidak dapat dimuat saat ini. (GEMINI_API_KEY belum dikonfigurasi)"

    prompt = f"Berikan 2 langkah praktis dan singkat untuk mengelola atau mendaur ulang sampah berjenis {label}. Balas dalam bahasa Indonesia yang ramah."
    try:
        # Menggunakan client baru dari google-genai SDK dengan model gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Tips daur ulang tidak dapat dimuat saat ini. Pastikan membuang sampah pada tempat yang sesuai."
