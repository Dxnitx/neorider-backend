from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import sensor, prediction, rides, health

app = FastAPI(
    title="NeoRider Backend",
    description="IoT-based intelligent motorcycle rider safety and training system",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sensor.router, prefix="/sensor", tags=["sensor"])
app.include_router(prediction.router, prefix="/predict", tags=["prediction"])
app.include_router(rides.router, prefix="/rides", tags=["rides"])
app.include_router(health.router, tags=["health"])

@app.on_event("startup")
async def startup_event():
    # Initialize Firebase on startup
    from app.firebase import initialize_firebase
    initialize_firebase()