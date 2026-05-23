from google import genai

from infrastructure.logging import logging
from utils.env import env_vars

gemini_client = genai.Client(api_key=env_vars.get("GEMINI_API_KEY"))


def _get_gemini_tips(label: str) -> str:
    """Meminta tips dari Gemini API berdasarkan prediksi sampah."""
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
