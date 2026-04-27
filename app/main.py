from fastapi import FastAPI
from app.core.database import Base, engine
from app.models.device import Device
from app.models.ride import Ride
from app.models.telemetry import Telemetry
from app.models.alert import Alert
from app.models.summary import RideSummary
from app.routers import device, ride, telemetry
    
Base.metadata.create_all(bind=engine)

app = FastAPI(title="NeoRider Backend")

app.include_router(device.router)
app.include_router(ride.router)
app.include_router(telemetry.router)

@app.get("/")
def root():
    return {"message": "NeoRider backend is running"}

from app.core.firebase import db

db.collection("test").add({
    "message": "NeoRider Firebase working!"
})