from fastapi import APIRouter, Request, HTTPException
from schemas.predict import HealthResponse

router = APIRouter()

def get_inference_engine(request: Request):
    return getattr(request.app.state, "inference_engine", None)

@router.get("/", tags=["Health"])
async def root():
    return {
        "message": "🌿 EcoWise Waste Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(request: Request):
    inference_engine = get_inference_engine(request)
    ready = inference_engine is not None
    return HealthResponse(
        status="ok" if ready else "model_not_loaded",
        model_ready=ready,
        classes=inference_engine.class_names if ready else [],
        version="1.0.0",
    )


@router.get("/classes", tags=["Info"])
async def get_classes(request: Request):
    inference_engine = get_inference_engine(request)
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {
        "classes": inference_engine.class_names,
        "descriptions": {
            "Anorganik": "Sampah tidak dapat terurai alami, mis. plastik, logam, kaca",
            "B3": "Bahan Berbahaya & Beracun, mis. baterai, obat, aerosol",
            "Organik": "Sampah dapat terurai alami, mis. sisa makanan, daun, kertas",
            "Non-Waste": "Objek umum/latar belakang yang bukan termasuk kategori limbah atau sampah",
        },
    }
