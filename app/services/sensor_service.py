from typing import List
import logging
from uuid import uuid4

from firebase_admin import firestore
from app.config import get_settings
from app.models.sensor import SensorReading, SensorReadingResponse

logger = logging.getLogger("neorider.sensor")

class SensorService:
    """Service for handling sensor reading operations."""

    def __init__(self, db: firestore.Client, enable_firestore_writes: bool | None = None):
        self.db = db
        self.enable_firestore_writes = (
            get_settings().enable_firestore_writes
            if enable_firestore_writes is None
            else enable_firestore_writes
        )

    def create_reading(self, reading: SensorReading) -> SensorReadingResponse:
        """Save a sensor reading to Firestore."""
        if not self.enable_firestore_writes:
            logger.info("[FIRESTORE] writes disabled")
            return SensorReadingResponse(id=str(uuid4()), **reading.model_dump())

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
