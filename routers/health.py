from fastapi import APIRouter

import core.model_state
from utils.response_schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    return {
        "message": "🌿 EcoWise Waste Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    ready = core.model_state.inference_engine is not None
    return HealthResponse(
        status="ok" if ready else "model_not_loaded",
        model_ready=ready,
        classes=core.model_state.inference_engine.class_names if ready else [],
        version="1.0.0",
    )
