from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: Dict[str, float]
    tips_daur_ulang: Optional[str] = None
    filename: Optional[str] = None
    latency_ms: Optional[float] = None


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    classes: List[str]
    version: str
