# app/models/live_prediction.py

from typing import Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict

from app.models.prediction import PredictionResponse


class LivePredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ride_id: str

    status: Literal[
        "collecting",
        "predicted",
    ]

    # Number of complete helmet/chest time steps.
    buffer_size: int
    helmet_buffer_size: int = 0
    chest_buffer_size: int = 0
    required_readings: int = 20
    paired_buffer_size: int = 0
    prediction_stride: int = 2
    prediction_latency_ms: Optional[float] = None

    prediction: Optional[PredictionResponse] = None
    helmet_prediction: Optional[Literal["SAFE", "RISK", "ACCIDENT"]] = None
    helmet_probabilities: Optional[Dict[str, float]] = None
    chest_prediction: Optional[Literal["SAFE", "RISK", "ACCIDENT"]] = None
    chest_probabilities: Optional[Dict[str, float]] = None
    fused_prediction: Optional[Literal["SAFE", "RISK", "ACCIDENT"]] = None
    fused_confidence: Optional[float] = None
    final_state: Optional[
        Literal["SAFE", "RISK", "ACCIDENT_PENDING", "ACCIDENT_CONFIRMED", "UNKNOWN"]
    ] = None
    impact_gate_passed: Optional[bool] = None
    is_helmet_stationary: Optional[bool] = None
    is_chest_stationary: Optional[bool] = None
    both_stationary: Optional[bool] = None
    accident_streak: Optional[int] = None
    accident_confirm_windows: int = 3
    motion_metrics: Optional[Dict[str, Dict[str, float]]] = None
