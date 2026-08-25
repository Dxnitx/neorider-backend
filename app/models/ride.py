from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime


class RideSummary(BaseModel):
    """Summary statistics for a ride."""
    
    safe_count: int = 0
    risky_count: int = 0
    accident_alerts: int = 0


class RideCreate(BaseModel):
    """Model for creating a new ride."""
    
    rider_name: str


class RideResponse(BaseModel):
    """Model for ride response."""
    
    id: str
    rider_name: str

    # Firestore timestamps
    start_time: datetime
    end_time: Optional[datetime] = None

    status: Literal["active", "completed"]

    total_readings: int = 0
    summary: RideSummary