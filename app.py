"""
app.py — EcoWise REST API (With Generative AI Integration)
==========================
FastAPI service untuk model klasifikasi sampah EcoWise.
Terima gambar → kembalikan label (Anorganik / B3 / Organik / Non-Waste) + confidence + Tips Daur Ulang.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.inference import EcoWiseInference
from utils.logger import logger
from routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load TensorFlow model
    logger.info(f"Memuat model dari: {settings.MODEL_PATH}")
    try:
        app.state.inference_engine = EcoWiseInference(
            settings.MODEL_PATH, settings.CLASS_NAMES_PATH
        )
        logger.info("Model TensorFlow berhasil dimuat")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        app.state.inference_engine = None

    yield

    # Shutdown
    logger.info("Menghentikan aplikasi EcoWise REST API...")


app = FastAPI(
    title="EcoWise Waste Classifier API",
    description=(
        "REST API untuk mengklasifikasikan jenis sampah: "
        "**Anorganik**, **B3**, **Organik**, atau **Non-Waste**. "
        "Powered by MobileNetV2 + TensorFlow & Gemini AI."
    ),
    version="1.0.0",
    contact={"name": "EcoWise Team"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(health.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
