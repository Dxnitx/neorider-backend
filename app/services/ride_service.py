# app/services/ride_service.py

from typing import List, Optional

from firebase_admin import firestore

from app.models.ride import RideCreate, RideResponse, RideSummary
from app.services.live_prediction_service import LivePredictionService


class RideService:
    """
    Service for handling ride session operations.
    """

    def __init__(self, db: firestore.Client):
        self.db = db

    def create_ride(self, ride: RideCreate) -> RideResponse:
        """
        Create a new ride session.
        """

        try:
            doc_ref = self.db.collection("rides").document()

            summary = RideSummary()

            data = {
                "rider_name": ride.rider_name,
                "start_time": firestore.SERVER_TIMESTAMP,
                "end_time": None,
                "status": "active",
                "total_readings": 0,
                "summary": summary.model_dump(),
            }

            doc_ref.set(data)

            # Read the saved document so Firestore timestamps
            # can be returned correctly.
            saved_doc = doc_ref.get()
            saved_data = saved_doc.to_dict()

            saved_data["id"] = saved_doc.id

            return RideResponse(**saved_data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to create ride: {str(e)}"
            )

    def get_ride(self, ride_id: str) -> Optional[RideResponse]:
        """
        Get one ride session by its ID.
        """

        try:
            doc = (
                self.db
                .collection("rides")
                .document(ride_id)
                .get()
            )

            if not doc.exists:
                return None

            data = doc.to_dict()
            data["id"] = doc.id

            return RideResponse(**data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve ride: {str(e)}"
            )

    def end_ride(self, ride_id: str) -> Optional[RideResponse]:
        """
        End an active ride, calculate its prediction summary,
        and clear its live prediction buffer.
        """

        try:
            doc_ref = (
                self.db
                .collection("rides")
                .document(ride_id)
            )

            doc = doc_ref.get()

            if not doc.exists:
                return None

            ride_data = doc.to_dict()

            if ride_data.get("status") == "completed":
                ride_data["id"] = doc.id
                return RideResponse(**ride_data)

            # Calculate summary using all predictions
            # created during this ride.
            summary = self._calculate_summary(ride_id)

            update_data = {
                "end_time": firestore.SERVER_TIMESTAMP,
                "status": "completed",
                "summary": summary.model_dump(),
            }

            doc_ref.update(update_data)

            # Remove the in-memory rolling sensor buffer.
            LivePredictionService(
                self.db
            ).clear_buffer(ride_id)

            updated_doc = doc_ref.get()
            updated_data = updated_doc.to_dict()
            updated_data["id"] = updated_doc.id

            return RideResponse(**updated_data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to end ride: {str(e)}"
            )

    def _calculate_summary(
        self,
        ride_id: str
    ) -> RideSummary:
        """
        Calculate ride summary statistics using saved predictions.
        """

        try:
            docs = (
                self.db
                .collection("predictions")
                .where("ride_id", "==", ride_id)
                .stream()
            )

            safe_count = 0
            risky_count = 0
            accident_alerts = 0

            for doc in docs:
                data = doc.to_dict()
                prediction = data.get(
                    "prediction",
                    "safe_riding"
                )

                if prediction == "safe_riding":
                    safe_count += 1

                elif prediction in [
                    "moderate_risk",
                    "high_risk",
                ]:
                    risky_count += 1

                elif prediction == "possible_accident":
                    accident_alerts += 1

            return RideSummary(
                safe_count=safe_count,
                risky_count=risky_count,
                accident_alerts=accident_alerts,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to calculate ride summary: {str(e)}"
            )

    def get_all_rides(self) -> List[RideResponse]:
        """
        Get all ride sessions ordered from newest to oldest.
        """

        try:
            docs = (
                self.db
                .collection("rides")
                .order_by(
                    "start_time",
                    direction=firestore.Query.DESCENDING,
                )
                .stream()
            )

            rides = []

            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id

                rides.append(
                    RideResponse(**data)
                )

            return rides

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve rides: {str(e)}"
            )