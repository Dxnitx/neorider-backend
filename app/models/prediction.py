from pydantic import BaseModel, ConfigDict, field_validator
from typing import Dict, List, Literal, Optional
from app.models.sensor import SensorReading


class PredictionRequest(BaseModel):
    ride_id: str
    sensor_window: List[SensorReading]

    @field_validator("sensor_window")
    @classmethod
    def validate_sensor_window(cls, v):
        if len(v) not in (20, 40):
            raise ValueError(
                "sensor_window must contain 20 Helmet readings, or the legacy "
                "40-reading payload (20 Helmet and 20 Chest)"
            )
        device_counts = {
            "helmet": sum(reading.device == "helmet" for reading in v),
            "chest": sum(reading.device == "chest" for reading in v),
        }
        if device_counts not in ({"helmet": 20, "chest": 0}, {"helmet": 20, "chest": 20}):
            raise ValueError(
                "sensor_window must contain exactly 20 Helmet readings; Chest is optional for legacy requests"
            )
        return v


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ride_id: str
    safety_class: Optional[Literal["SAFE", "RISK", "ACCIDENT"]] = None
    class_id: Optional[int] = None
    prediction: Literal[
        "safe_riding",
        "moderate_risk",
        "high_risk",
        "possible_accident",
    ]
    confidence: float
    timestamp: str
    model_used: str
    # Additive/optional so existing clients and stored history remain compatible.
    probabilities: Optional[Dict[str, float]] = None
    window_size: Optional[int] = None
    sample_rate_hz: Optional[int] = None
    feature_count: Optional[int] = None
