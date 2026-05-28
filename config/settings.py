import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Helper function to get clean environment variables (without quotes)
def get_clean_env(key: str, default: str = None) -> str:
    val = os.getenv(key, default)
    if val:
        val = val.strip('"' + "'")
    return val


MODEL_PATH = get_clean_env("MODEL_PATH", "models/model_ecowise_production.keras")
CLASS_NAMES_PATH = get_clean_env("CLASS_NAMES_PATH", "class_names.json")
raw_max_size = get_clean_env("MAX_FILE_SIZE_MB", "10")
MAX_FILE_SIZE_MB = int(raw_max_size) if raw_max_size.isdigit() else 10

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
GEMINI_API_KEY = get_clean_env("GEMINI_API_KEY")
