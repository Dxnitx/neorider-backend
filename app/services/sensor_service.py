from typing import List
from firebase_admin import firestore
from app.models.sensor import SensorReading, SensorReadingResponse

class SensorService:
    """Service for handling sensor reading operations."""

    def __init__(self, db: firestore.Client):
        self.db = db

    def create_reading(self, reading: SensorReading) -> SensorReadingResponse:
        """Save a sensor reading to Firestore."""
        doc_ref = self.db.collection('sensor_readings').document()
        data = reading.model_dump()
        data['server_timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref.set(data)
        return SensorReadingResponse(id=doc_ref.id, **reading.model_dump())

    def get_readings_by_ride(self, ride_id: str) -> List[SensorReading]:
        """Get all sensor readings for a ride."""
        docs = self.db.collection('sensor_readings').where('ride_id', '==', ride_id).order_by('timestamp').stream()
        readings = []
        for doc in docs:
            data = doc.to_dict()
            readings.append(SensorReading(**data))
        return readings
