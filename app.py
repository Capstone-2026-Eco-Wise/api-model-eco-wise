"""
app.py — EcoWise REST API (With Generative AI Integration)
==========================
FastAPI service untuk model klasifikasi sampah EcoWise.
Terima gambar → kembalikan label (Anorganik / B3 / Organik / Non-Waste) + confidence + Tips Daur Ulang.
"""

import os
import time
import logging
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import generativeai as genai

# Import inference engine 
from inference import EcoWiseInference

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("ecowise-api")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model_ecowise_production.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ── Konfigurasi Gemini API  ───────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "MASUKKAN API NYA DISINI YAA")
genai.configure(api_key=GEMINI_API_KEY)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="EcoWise Waste Classifier API",
    description=(
        "REST API untuk mengklasifikasikan jenis sampah: "
        "**Anorganik**, **B3**, **Organik**, atau **Non-Waste**. "
        "Powered by MobileNetV2 + TensorFlow & Gemini AI."
    ),
    version="1.0.0",
    contact={"name": "EcoWise Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model saat startup ───────────────────────────────────────────────────
inference_engine: EcoWiseInference = None


@app.on_event("startup")
async def startup_event():
    global inference_engine
    logger.info(f"Memuat model dari: {MODEL_PATH}")
    try:
        inference_engine = EcoWiseInference(MODEL_PATH, CLASS_NAMES_PATH)
        logger.info("Model TensorFlow berhasil dimuat")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        raise RuntimeError(f"Model tidak dapat dimuat: {e}")


# ── Response schemas ──────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: dict
    tips_daur_ulang: str | None = None  
    filename: str | None = None
    latency_ms: float | None = None


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    classes: List[str]
    version: str


# ── Helper ────────────────────────────────────────────────────────────────────
def _validate_file(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {file.content_type}. Gunakan: {ALLOWED_TYPES}",
        )


def _get_gemini_tips(label: str) -> str:
    """Meminta tips dari Gemini API berdasarkan prediksi sampah."""
    # Jika objek bukan sampah, langsung return pesan bypass
    if label == "Non-Waste":
        return "Objek ini bukan merupakan kategori sampah. Tidak ada tips daur ulang yang tersedia."

    prompt = f"Berikan 2 langkah praktis dan singkat untuk mengelola atau mendaur ulang sampah berjenis {label}. Balas dalam bahasa Indonesia yang ramah."
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash") 
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Tips daur ulang tidak dapat dimuat saat ini. Pastikan membuang sampah pada tempat yang sesuai."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "🌿 EcoWise Waste Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    ready = inference_engine is not None
    return HealthResponse(
        status="ok" if ready else "model_not_loaded",
        model_ready=ready,
        classes=inference_engine.class_names if ready else [],
        version="1.0.0",
    )


@app.get("/classes", tags=["Info"])
async def get_classes():
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {
        "classes": inference_engine.class_names,
        # Menambahkan deskripsi untuk kelas Non-Waste
        "descriptions": {
            "Anorganik": "Sampah tidak dapat terurai alami, mis. plastik, logam, kaca",
            "B3": "Bahan Berbahaya & Beracun, mis. baterai, obat, aerosol",
            "Organik": "Sampah dapat terurai alami, mis. sisa makanan, daun, kertas",
            "Non-Waste": "Objek umum/latar belakang yang bukan termasuk kategori limbah atau sampah",
        },
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(file: UploadFile = File(...)):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    _validate_file(file)
    img_bytes = await file.read()

    if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar. Maks {MAX_FILE_SIZE_MB} MB.",
        )

    t0 = time.perf_counter()
    try:
        # Prediksi TensorFlow
        result = inference_engine.predict_from_bytes(img_bytes)
        # Prediksi Gemini 
        tips = _get_gemini_tips(result["label"])

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference gagal: {str(e)}")

    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"predict | file={file.filename} | label={result['label']} "
        f"| conf={result['confidence']:.3f} | {latency}ms"
    )

    return PredictionResult(
        label=result["label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        tips_daur_ulang=tips,
        filename=file.filename,
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maks 20 gambar per request")

    t_batch_start = time.perf_counter()
    results = []

    for file in files:
        _validate_file(file)
        img_bytes = await file.read()

        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' terlalu besar.",
            )

        t0 = time.perf_counter()
        try:
            result = inference_engine.predict_from_bytes(img_bytes)
            tips = _get_gemini_tips(result["label"])
        except Exception as e:
            logger.error(f"Batch inference error [{file.filename}]: {e}")
            raise HTTPException(
                status_code=500, detail=f"Gagal proses {file.filename}: {e}"
            )

        latency = round((time.perf_counter() - t0) * 1000, 2)
        results.append(
            PredictionResult(
                label=result["label"],
                confidence=result["confidence"],
                all_scores=result["all_scores"],
                tips_daur_ulang=tips,
                filename=file.filename,
                latency_ms=latency,
            )
        )

    total_latency = round((time.perf_counter() - t_batch_start) * 1000, 2)
    logger.info(f"batch predict | files={len(files)} | total={total_latency}ms")

    return BatchPredictionResult(
        results=results,
        total=len(results),
        latency_ms=total_latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)