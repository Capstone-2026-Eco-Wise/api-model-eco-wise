import os

from dotenv import load_dotenv

load_dotenv()

env_vars = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "MODEL_PATH": os.getenv("MODEL_PATH"),
    "CLASS_NAMES_PATH": os.getenv("CLASS_NAMES_PATH"),
    "MAX_FILE_SIZE_MB": os.getenv("MAX_FILE_SIZE_MB"),
    "RESTFUL_API_URL": os.getenv("RESTFUL_API_URL"),
}
