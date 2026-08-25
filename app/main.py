# app/main.py

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import health, ml, prediction, rides, sensor


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application services when the backend starts.
    """

    from app.services.mmc1_model_service import log_startup_status

    log_startup_status()
    if os.getenv("TESTING", "").lower() != "true":
        from app.firebase import initialize_firebase

        initialize_firebase()

    yield


app = FastAPI(
    title="NeoRider Backend",
    description=(
        "IoT-based intelligent motorcycle rider "
        "safety and training system"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(
    ml.router,
    prefix="/ml",
    tags=["ML"],
)

app.include_router(
    sensor.router,
    prefix="/sensor",
    tags=["Sensor"],
)

app.include_router(
    prediction.router,
    prefix="/predict",
    tags=["Prediction"],
)

app.include_router(
    rides.router,
    prefix="/rides",
    tags=["Rides"],
)

app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    NeoRider backend root endpoint.
    """

    return {
        "message": "NeoRider Backend is running",
        "version": "1.0.0",
        "docs": "/docs",
        "live_sensor_endpoint": "/sensor/live",
        "manual_prediction_endpoint": "/predict",
        "health_endpoint": "/health",
    }
