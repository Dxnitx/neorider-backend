from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.ride import RideCreate, RideResponse
from app.services.ride_service import RideService
from app.firebase import get_firestore_client

router = APIRouter()

@router.post("", status_code=201, response_model=RideResponse)
async def create_ride(
    ride: RideCreate,
    db=Depends(get_firestore_client)
):
    """Create a new ride session."""
    service = RideService(db)
    try:
        result = service.create_ride(ride)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ride: {str(e)}")

@router.get("/{ride_id}", response_model=RideResponse)
async def get_ride(
    ride_id: str,
    db=Depends(get_firestore_client)
):
    """Get ride session details."""
    service = RideService(db)
    try:
        ride = service.get_ride(ride_id)
        if not ride:
            raise HTTPException(status_code=404, detail="Ride not found")
        return ride
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ride: {str(e)}")

@router.patch("/{ride_id}/end", response_model=RideResponse)
async def end_ride(
    ride_id: str,
    db=Depends(get_firestore_client)
):
    """End a ride session."""
    service = RideService(db)
    try:
        result = service.end_ride(ride_id)
        if not result:
            raise HTTPException(status_code=404, detail="Ride not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end ride: {str(e)}")

@router.get("", response_model=List[RideResponse])
async def get_rides(
    db=Depends(get_firestore_client)
):
    """Get all ride sessions."""
    service = RideService(db)
    try:
        rides = service.get_all_rides()
        return rides
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rides: {str(e)}")