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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

env_model = get_clean_env("MODEL_PATH")
if env_model:
    MODEL_PATH = env_model if os.path.isabs(env_model) else os.path.abspath(os.path.join(ROOT_DIR, env_model))
else:
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_ecowise_production.keras")

env_class = get_clean_env("CLASS_NAMES_PATH")
if env_class:
    CLASS_NAMES_PATH = env_class if os.path.isabs(env_class) else os.path.abspath(os.path.join(ROOT_DIR, env_class))
else:
    CLASS_NAMES_PATH = os.path.join(ROOT_DIR, "class_names.json")

raw_max_size = get_clean_env("MAX_FILE_SIZE_MB", "5")
MAX_FILE_SIZE_MB = int(raw_max_size) if raw_max_size.isdigit() else 5
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
GEMINI_API_KEY = get_clean_env("GEMINI_API_KEY")
