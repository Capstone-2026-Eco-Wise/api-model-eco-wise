from pydantic import BaseModel
from typing import List


class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: dict
    tips_daur_ulang: str | None = None  # Ditambahkan untuk menampung hasil Gemini
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
