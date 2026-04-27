from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.firebase import get_firestore_client

router = APIRouter()

@router.post("", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
    db=Depends(get_firestore_client)
):
    """Create a prediction from sensor window."""
    service = PredictionService(db)
    try:
        result = service.create_prediction(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create prediction: {str(e)}")

@router.get("/history/{ride_id}")
async def get_prediction_history(
    ride_id: str,
    db=Depends(get_firestore_client)
) -> List[PredictionResponse]:
    """Get all predictions for a ride."""
    service = PredictionService(db)
    try:
        predictions = service.get_predictions_by_ride(ride_id)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve predictions: {str(e)}")