from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.core.database import get_db
from app.models.device import Device

router = APIRouter(prefix="/device", tags=["Device"])


@router.post("/register")
def register_device(
    user_id: int,
    device_name: str,
    device_type: str,
    device_identifier: str,
    db: Session = Depends(get_db)
):
    try:
        new_device = Device(
            user_id=user_id,
            device_name=device_name,
            device_type=device_type.strip(),  # remove extra spaces
            device_identifier=device_identifier.strip()
        )

        db.add(new_device)
        db.commit()
        db.refresh(new_device)

        return {"message": "Device registered", "device_id": new_device.id}

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Device already exists")

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))