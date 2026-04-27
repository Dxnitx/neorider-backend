from typing import List, Optional
from datetime import datetime
from firebase_admin import firestore
from app.models.ride import RideCreate, RideResponse, RideSummary

class RideService:
    """Service for handling ride session operations."""

    def __init__(self, db: firestore.Client):
        self.db = db

    def create_ride(self, ride: RideCreate) -> RideResponse:
        """Create a new ride session."""
        doc_ref = self.db.collection('rides').document()
        data = {
            'rider_name': ride.rider_name,
            'start_time': firestore.SERVER_TIMESTAMP,
            'end_time': None,
            'status': 'active',
            'total_readings': 0,
            'summary': RideSummary().dict()
        }
        doc_ref.set(data)
        return RideResponse(
            id=doc_ref.id,
            rider_name=ride.rider_name,
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            status='active',
            total_readings=0,
            summary=RideSummary()
        )

    def get_ride(self, ride_id: str) -> Optional[RideResponse]:
        """Get ride session details."""
        doc = self.db.collection('rides').document(ride_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data['id'] = doc.id
        return RideResponse(**data)

    def end_ride(self, ride_id: str) -> Optional[RideResponse]:
        """End a ride session and update summary."""
        doc_ref = self.db.collection('rides').document(ride_id)
        doc = doc_ref.get()
        if not doc.exists:
            return None
        
        # Calculate summary from predictions
        summary = self._calculate_summary(ride_id)
        
        # Update ride
        update_data = {
            'end_time': firestore.SERVER_TIMESTAMP,
            'status': 'completed',
            'summary': summary.dict()
        }
        doc_ref.update(update_data)
        
        # Return updated ride
        updated_doc = doc_ref.get()
        data = updated_doc.to_dict()
        data['id'] = updated_doc.id
        return RideResponse(**data)

    def _calculate_summary(self, ride_id: str) -> RideSummary:
        """Calculate summary statistics from predictions."""
        docs = self.db.collection('predictions').where('ride_id', '==', ride_id).stream()
        safe_count = 0
        risky_count = 0
        accident_alerts = 0
        
        for doc in docs:
            pred = doc.to_dict()['prediction']
            if pred == 'safe_riding':
                safe_count += 1
            elif pred in ['risky_behavior', 'unstable_movement', 'sudden_motion']:
                risky_count += 1
            elif pred == 'possible_accident':
                accident_alerts += 1
        
        return RideSummary(
            safe_count=safe_count,
            risky_count=risky_count,
            accident_alerts=accident_alerts
        )

    def get_all_rides(self) -> List[RideResponse]:
        """Get all ride sessions."""
        docs = self.db.collection('rides').order_by('start_time', direction=firestore.Query.DESCENDING).stream()
        rides = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            rides.append(RideResponse(**data))
        return rides