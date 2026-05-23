import logging
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import model state
import core.model_state
from inference import EcoWiseInference
from infrastructure.logging import logging as custom_logging

# Import routers
from routers import health, info, predict
from utils.env import env_vars

load_dotenv()
logger = custom_logging.getLogger(__name__)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
MODEL_PATH = env_vars.get("MODEL_PATH")
CLASS_NAMES_PATH = env_vars.get("CLASS_NAMES_PATH")
ORIGIN_ALLOWED = env_vars.get("RESRESTFUL_API_URL")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="EcoWise Waste Classifier API",
    description=(
        "REST API untuk mengklasifikasikan jenis sampah: "
        "**Anorganik**, **B3** (Bahan Berbahaya & Beracun), atau **Organik**. "
        "Powered by MobileNetV2 + TensorFlow & Gemini AI."
    ),
    version="1.0.0",
    contact={"name": "EcoWise Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ORIGIN_ALLOWED] if ORIGIN_ALLOWED else ["*"],
)

# ── Mendaftarkan Routers ──────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(info.router)
app.include_router(predict.router)


# ── Load model saat startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info(f"Memuat model dari: {MODEL_PATH}")
    try:
        core.model_state.inference_engine = EcoWiseInference(
            MODEL_PATH, CLASS_NAMES_PATH
        )
        logger.info("Model TensorFlow berhasil dimuat")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        raise RuntimeError(f"Model tidak dapat dimuat: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
