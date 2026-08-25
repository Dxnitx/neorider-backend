from fastapi import APIRouter

from app.services.mmc1_model_service import model_status

router = APIRouter()


@router.get("/model-status")
async def get_model_status():
    return model_status()
