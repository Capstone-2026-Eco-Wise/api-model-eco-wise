from google import genai

from infrastructure.logging import logging
from utils.env import env_vars

gemini_client = genai.Client(api_key=env_vars.get("GEMINI_API_KEY"))
logger = logging.getLogger(__name__)


def _get_gemini_tips(label: str) -> str:
    """Meminta tips dari Gemini API berdasarkan prediksi sampah."""

    if label == "Non-Waste":
        return "Objek ini bukan merupakan kategori sampah. Tidak ada tips daur ulang yang tersedia."

    prompt = f"Berikan 2 langkah praktis dan singkat untuk mengelola atau mendaur ulang sampah berjenis {label}. Balas dalam bahasa Indonesia yang ramah."
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Tips daur ulang tidak dapat dimuat saat ini. Pastikan membuang sampah pada tempat yang sesuai."
