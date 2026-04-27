from fastapi import APIRouter, Depends
from app.config import get_settings

router = APIRouter()

@router.get("/health")
async def health_check(settings=Depends(get_settings)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": settings.app_version
    }