from pydantic import BaseModel, validator
from typing import List, Literal
from app.models.sensor import SensorReading

class PredictionRequest(BaseModel):
    """Model for prediction request."""
    ride_id: str
    sensor_window: List[SensorReading]

    @validator('sensor_window')
    def validate_sensor_window(cls, v):
        if len(v) != 20:
            raise ValueError('sensor_window must contain exactly 20 sensor readings')
        return v

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    ride_id: str
    prediction: Literal["safe_riding", "unstable_movement", "sudden_motion", "risky_behavior", "possible_accident"]
    confidence: float
    timestamp: str  # ISO 8601 string
    model_used: str