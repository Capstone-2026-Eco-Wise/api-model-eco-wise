import time
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import core.model_state
from infrastructure.logging import logging
from utils.env import env_vars
from utils.gemini_tips import _get_gemini_tips
from utils.response_schemas import BatchPredictionResult, PredictionResult
from utils.validate_file import _validate_file

router = APIRouter(tags=["Prediction"])
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(env_vars.get("MAX_FILE_SIZE_MB", 10))


@router.post("/predict", response_model=PredictionResult)
async def predict_single(file: UploadFile = File(...), token: int = Form(0)):
    if not core.model_state.inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")

    _validate_file(file)
    img_bytes = await file.read()

    logger.info(f"TOKEN: {token}")

    if len(img_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar. Maks {MAX_FILE_SIZE_MB} MB.",
        )

    t0 = time.perf_counter()

    try:
        result = core.model_state.inference_engine.predict_from_bytes(img_bytes)

        if token != 0:
            tips = _get_gemini_tips(result["label"])
        else:
            tips = "Token Anda sudah habis. Silahkan tunggu untuk mendapatkan tips daur ulang."
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


@router.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(files: List[UploadFile] = File(...)):
    if not core.model_state.inference_engine:
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
            result = core.model_state.inference_engine.predict_from_bytes(img_bytes)
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
