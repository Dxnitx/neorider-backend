from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.ride import Ride

router = APIRouter(prefix="/ride", tags=["Ride"])


@router.post("/start")
def start_ride(user_id: int, device_id: int, db: Session = Depends(get_db)):
    ride = Ride(
        user_id=user_id,
        device_id=device_id
    )

    db.add(ride)
    db.commit()
    db.refresh(ride)

    return {"message": "Ride started", "ride_id": ride.id}