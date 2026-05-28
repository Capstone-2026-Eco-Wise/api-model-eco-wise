from fastapi import UploadFile, HTTPException
from config.settings import ALLOWED_TYPES

def validate_file(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {file.content_type}. Gunakan: {ALLOWED_TYPES}",
        )
