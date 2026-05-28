import time
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from config.settings import MAX_FILE_SIZE_MB
from core.gemini import get_gemini_tips
from schemas.predict import PredictionResult, BatchPredictionResult
from utils.validators import validate_file
from utils.logger import logger

router = APIRouter()

def get_inference_engine(request: Request):
    return getattr(request.app.state, "inference_engine", None)

@router.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(request: Request, file: UploadFile = File(...)):
    inference_engine = get_inference_engine(request)
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    validate_file(file)
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
        tips = get_gemini_tips(result["label"])

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


@router.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    inference_engine = get_inference_engine(request)
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maks 20 gambar per request")

    t_batch_start = time.perf_counter()
    results = []

    for file in files:
        validate_file(file)
        img_bytes = await file.read()

        if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' terlalu besar.",
            )

        t0 = time.perf_counter()
        try:
            result = inference_engine.predict_from_bytes(img_bytes)
            tips = get_gemini_tips(result["label"])
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
