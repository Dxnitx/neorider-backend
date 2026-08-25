# app/services/prediction_service.py

from datetime import datetime
from typing import List
from firebase_admin import firestore

from app.models.prediction import (
    PredictionRequest,
    PredictionResponse
)

from app.services.mmc1_model_service import predict_safety


class PredictionService:
    """
    Service for handling prediction operations.
    """

    def __init__(self, db: firestore.Client):
        self.db = db

    def create_prediction(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:

        try:
            # ------------------------------------------
            # STEP 1 - Extract features from sensor data
            # ------------------------------------------

            helmet_window = [reading for reading in request.sensor_window if reading.device == "helmet"]
            prediction_result = predict_safety(helmet_window)

            # ------------------------------------------
            # STEP 3 - Create API response
            # ------------------------------------------

            response = PredictionResponse(
                ride_id=request.ride_id,
                safety_class=prediction_result["safety_class"],
                class_id=prediction_result["class_id"],
                prediction=prediction_result["prediction"],
                confidence=prediction_result["confidence"],
                timestamp=datetime.utcnow().isoformat(),
                model_used=prediction_result["model_used"],
                probabilities=prediction_result["probabilities"],
                window_size=prediction_result["window_size"],
                sample_rate_hz=prediction_result["sample_rate_hz"],
                feature_count=prediction_result["feature_count"],
            )

            # ------------------------------------------
            # STEP 4 - Save prediction to Firestore
            # ------------------------------------------

            doc_ref = self.db.collection(
                "predictions"
            ).document()

            firestore_data = {
                "ride_id": response.ride_id,
                "safety_class": response.safety_class,
                "class_id": response.class_id,
                "prediction": response.prediction,
                "confidence": response.confidence,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "model_used": response.model_used,
                "probabilities": response.probabilities,
            }

            doc_ref.set(firestore_data)

            # ------------------------------------------
            # STEP 5 - Optional alert generation
            # ------------------------------------------

            risky_behaviors = [
                "high_risk",
                "possible_accident"
            ]

            if (
                response.prediction in risky_behaviors
                and response.confidence >= 0.75
            ):

                alert_data = {
                    "ride_id": response.ride_id,
                    "alert_type": "risky_behavior",
                    "behavior": response.prediction,
                    "confidence": response.confidence,
                    "message": f"Risky rider behavior detected: {response.prediction}",
                    "timestamp": firestore.SERVER_TIMESTAMP
                }

                self.db.collection(
                    "alerts"
                ).add(alert_data)

            # ------------------------------------------
            # Return final response
            # ------------------------------------------

            return response

        except Exception as e:
            raise RuntimeError(
                f"Prediction creation failed: {str(e)}"
            )

    def get_predictions_by_ride(self, ride_id: str) -> List[PredictionResponse]:
        try:
            docs = (
                self.db.collection("predictions")
                .where("ride_id", "==", ride_id)
                .order_by("timestamp")
                .stream()
            )

            predictions = []

            label_map = {
                "cruise": "safe_riding",
                "wait": "safe_riding",
                "traffic": "moderate_risk",
                "overtake": "high_risk",
                "fun": "moderate_risk",
                "accident": "possible_accident",
                "possible_accident": "possible_accident",
                "safe_riding": "safe_riding",
                "moderate_risk": "moderate_risk",
                "high_risk": "high_risk",
            }

            for doc in docs:
                data = doc.to_dict()

                prediction = data.get("prediction", "safe_riding")
                prediction = label_map.get(prediction, "safe_riding")

                timestamp = data.get("timestamp")
                if timestamp and hasattr(timestamp, "isoformat"):
                    timestamp = timestamp.isoformat()
                elif timestamp is None:
                    timestamp = ""

                predictions.append(
                    PredictionResponse(
                        ride_id=data.get("ride_id", ride_id),
                        safety_class=data.get("safety_class"),
                        class_id=data.get("class_id"),
                        prediction=prediction,
                        confidence=float(data.get("confidence", 0)),
                        timestamp=timestamp,
                        model_used=data.get("model_used", "unknown"),
                    )
                )

            return predictions

        except Exception as e:
            raise Exception(f"Failed to retrieve predictions: {str(e)}")
