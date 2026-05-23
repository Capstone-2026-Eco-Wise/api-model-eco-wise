from fastapi import UploadFile

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _validate_file(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {file.content_type}. "
            f"Gunakan: {ALLOWED_TYPES}",
        )
