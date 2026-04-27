from typing import List, Dict
from datetime import datetime
from firebase_admin import firestore
from app.models.prediction import PredictionRequest, PredictionResponse
from app.utils.feature_extractor import extract_features

class PredictionService:
    """Service for handling prediction operations."""

    def __init__(self, db: firestore.Client):
        self.db = db

    def create_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Create a prediction from sensor window."""
        # Extract features from sensor window
        features = extract_features(request.sensor_window)
        
        # Get prediction from ML model (placeholder)
        prediction_result = self.predict_behavior(features)
        
        # Create response
        response = PredictionResponse(
            ride_id=request.ride_id,
            prediction=prediction_result['prediction'],
            confidence=prediction_result['confidence'],
            timestamp=datetime.utcnow().isoformat(),
            model_used=prediction_result['model_used']
        )
        
        # Save to Firestore
        doc_ref = self.db.collection('predictions').document()
        data = response.dict()
        data['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref.set(data)
        
        return response

    def predict_behavior(self, features: Dict[str, float]) -> Dict:
        """ML placeholder function.
        
        TODO: Replace with trained ML model in Phase 3
        Currently ignores input features and returns hardcoded result.
        """
        # Hardcoded mock result
        return {
            "prediction": "safe_riding",
            "confidence": 0.91,
            "model_used": "placeholder_v0"
        }

    def get_predictions_by_ride(self, ride_id: str) -> List[PredictionResponse]:
        """Get all predictions for a ride."""
        docs = self.db.collection('predictions').where('ride_id', '==', ride_id).order_by('timestamp').stream()
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            predictions.append(PredictionResponse(**data))
        return predictions