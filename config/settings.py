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


dir_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(dir_path, "../models/model_ecowise_production.keras")
CLASS_NAMES_PATH = os.path.join(dir_path, "../class_names.json")
raw_max_size = get_clean_env("MAX_FILE_SIZE_MB", "10")
MAX_FILE_SIZE_MB = int(raw_max_size) if raw_max_size.isdigit() else 5

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
GEMINI_API_KEY = get_clean_env("GEMINI_API_KEY")
