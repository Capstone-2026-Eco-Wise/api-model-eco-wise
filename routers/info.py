from fastapi import APIRouter, HTTPException

import core.model_state

router = APIRouter(tags=["Info"])


@router.get("/classes")
async def get_classes():
    if not core.model_state.inference_engine:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {
        "classes": core.model_state.inference_engine.class_names,
        "descriptions": {
            "Anorganik": "Sampah tidak dapat terurai alami, mis. plastik, logam, kaca",
            "B3": "Bahan Berbahaya & Beracun, mis. baterai, obat, aerosol",
            "Organik": "Sampah dapat terurai alami, mis. sisa makanan, daun, kertas",
        },
    }
