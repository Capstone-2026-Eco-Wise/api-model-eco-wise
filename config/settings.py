import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model_ecowise_production.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "MASUKKAN API NYA DISINI YAA")
