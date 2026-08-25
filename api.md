# NeoRider Backend API Documentation

## Overview
NeoRider is an IoT-based intelligent motorcycle rider safety system. This API handles sensor data ingestion, ride session management, and behavior prediction using machine learning.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required (development mode).

## Endpoints

### Health Check
#### GET /health
Returns the health status of the API.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Sensor Data
#### POST /sensor/reading
Submit a single sensor reading from helmet or chest IMU.

**Request Body:**
```json
{
  "ride_id": "string",
  "device": "helmet" | "chest",
  "timestamp": "ISO 8601 string",
  "accel_x": "float",
  "accel_y": "float",
  "accel_z": "float",
  "gyro_x": "float",
  "gyro_y": "float",
  "gyro_z": "float",
  "pitch": "float",
  "roll": "float",
  "yaw": "float"
}
```

**Response (201):**
```json
{
  "id": "firestore_document_id",
  "ride_id": "string",
  "device": "helmet",
  "timestamp": "ISO 8601 string",
  "accel_x": 1.0,
  "accel_y": 2.0,
  "accel_z": 3.0,
  "gyro_x": 0.1,
  "gyro_y": 0.2,
  "gyro_z": 0.3,
  "pitch": 10.0,
  "roll": 20.0,
  "yaw": 30.0
}
```

#### GET /sensor/readings/{ride_id}
Retrieve all sensor readings for a specific ride.

**Parameters:**
- `ride_id` (path): Ride session identifier

**Response (200):**
```json
[
  {
    "ride_id": "string",
    "device": "helmet",
    "timestamp": "ISO 8601 string",
    "accel_x": 1.0,
    "accel_y": 2.0,
    "accel_z": 3.0,
    "gyro_x": 0.1,
    "gyro_y": 0.2,
    "gyro_z": 0.3,
    "pitch": 10.0,
    "roll": 20.0,
    "yaw": 30.0
  }
]
```

### Prediction
#### POST /predict
Generate behavior prediction from a 20-sensor reading window.

**Request Body:**
```json
{
  "ride_id": "string",
  "sensor_window": [
    {
      "ride_id": "string",
      "device": "helmet",
      "timestamp": "ISO 8601 string",
      "accel_x": 1.0,
      "accel_y": 2.0,
      "accel_z": 3.0,
      "gyro_x": 0.1,
      "gyro_y": 0.2,
      "gyro_z": 0.3,
      "pitch": 10.0,
      "roll": 20.0,
      "yaw": 30.0
    },
    // ... 19 more readings
  ]
}
```

**Response (200):**
```json
{
  "ride_id": "string",
  "prediction": "safe_riding",
  "confidence": 0.91,
  "timestamp": "ISO 8601 string",
  "model_used": "LightGBM_tuned_real_dataset"
}
```

**Prediction Classes:**
- `safe_riding`
- `moderate_risk`
- `high_risk`
- `possible_accident`

#### GET /predict/history/{ride_id}
Get prediction history for a ride.

**Parameters:**
- `ride_id` (path): Ride session identifier

**Response (200):**
```json
[
  {
    "ride_id": "string",
    "prediction": "safe_riding",
    "confidence": 0.91,
    "timestamp": "ISO 8601 string",
    "model_used": "LightGBM_tuned_real_dataset"
  }
]
```

### Ride Management
#### POST /rides
Create a new ride session.

**Request Body:**
```json
{
  "rider_name": "string"
}
```

**Response (201):**
```json
{
  "id": "firestore_document_id",
  "rider_name": "string",
  "start_time": "ISO 8601 string",
  "end_time": null,
  "status": "active",
  "total_readings": 0,
  "summary": {
    "safe_count": 0,
    "risky_count": 0,
    "accident_alerts": 0
  }
}
```

#### GET /rides/{ride_id}
Get details of a specific ride.

**Parameters:**
- `ride_id` (path): Ride session identifier

**Response (200):**
```json
{
  "id": "string",
  "rider_name": "string",
  "start_time": "ISO 8601 string",
  "end_time": "ISO 8601 string",
  "status": "completed",
  "total_readings": 100,
  "summary": {
    "safe_count": 80,
    "risky_count": 15,
    "accident_alerts": 5
  }
}
```

#### PATCH /rides/{ride_id}/end
End a ride session and calculate final summary.

**Parameters:**
- `ride_id` (path): Ride session identifier

**Response (200):**
```json
{
  "id": "string",
  "rider_name": "string",
  "start_time": "ISO 8601 string",
  "end_time": "ISO 8601 string",
  "status": "completed",
  "total_readings": 100,
  "summary": {
    "safe_count": 80,
    "risky_count": 15,
    "accident_alerts": 5
  }
}
```

#### GET /rides
List all ride sessions.

**Response (200):**
```json
[
  {
    "id": "string",
    "rider_name": "string",
    "start_time": "ISO 8601 string",
    "end_time": "ISO 8601 string",
    "status": "completed",
    "total_readings": 100,
    "summary": {
      "safe_count": 80,
      "risky_count": 15,
      "accident_alerts": 5
    }
  }
]
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error message"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 422 Unprocessable Entity
```json
{
  "detail": "Validation error details"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

## Data Structures and Algorithms

### Data Structures

#### Sensor Reading
- **Type:** Dictionary/JSON Object
- **Fields:** 12 attributes (ride_id, device, timestamp, 9 sensor values)
- **Storage:** Firestore document
- **Indexing:** ride_id for efficient querying

#### Prediction Result
- **Type:** Dictionary/JSON Object
- **Fields:** 5 attributes (ride_id, prediction, confidence, timestamp, model_used)
- **Storage:** Firestore document
- **Indexing:** ride_id for aggregation

#### Ride Session
- **Type:** Dictionary/JSON Object
- **Fields:** 7 attributes including nested summary object
- **Storage:** Firestore document
- **Indexing:** start_time for chronological ordering

#### Feature Vector
- **Type:** Dictionary of 27 float values
- **Structure:** {channel_stat: value} where stat is mean/std/max
- **Usage:** ML model input
- **Generation:** Statistical computation from sensor windows

### Algorithms

#### Feature Extraction
- **Input:** List of 20 sensor reading dictionaries
- **Process:**
  1. Extract 9 sensor channels from each reading
  2. Compute the 156 statistical features used during model training
  3. Return flattened dictionary aligned with the saved scaler feature names
- **Complexity:** O(n) where n=20 (constant time)
- **Libraries:** NumPy, Pandas

#### Prediction Classification
- **Current:** LightGBM inference using saved model artifacts
- **Input:** 156-dimensional feature vector
- **Output:** Risk label + confidence score

#### Summary Calculation
- **Input:** Collection of predictions for a ride
- **Process:** Count occurrences of each prediction class
- **Complexity:** O(m) where m=number of predictions
- **Categories:** safe_riding, moderate_risk, high_risk, possible_accident

## Architecture

### Application Architecture
- **Framework:** FastAPI (ASGI web framework)
- **Language:** Python 3.11+
- **Style:** RESTful API with async endpoints
- **Structure:** Layered architecture (Routers → Services → Firebase)

### Data Architecture
- **Database:** Firebase Firestore (NoSQL document database)
- **Collections:**
  - `sensor_readings`: Individual IMU readings
  - `predictions`: ML classification results
  - `rides`: Session metadata and summaries
- **Schema:** Document-based with nested objects
- **Indexing:** Automatic on document IDs, manual on ride_id fields

### Service Architecture
- **Dependency Injection:** FastAPI Depends() for database connections
- **Business Logic:** Separated into service classes
- **Error Handling:** HTTPException with appropriate status codes
- **Validation:** Pydantic models for request/response validation

### Development Architecture
- **Environment:** dotenv for configuration management
- **Testing:** pytest with mocked Firebase calls
- **Documentation:** Auto-generated OpenAPI/Swagger UI
- **Version Control:** Git with comprehensive .gitignore

### Deployment Architecture
- **Server:** Uvicorn ASGI server
- **CORS:** Enabled for cross-origin requests
- **Health Checks:** Dedicated endpoint for monitoring
- **Logging:** Standard Python logging (configurable)

### Security Architecture
- **Authentication:** None (development phase)
- **Authorization:** None (development phase)
- **Data Protection:** Firebase service account keys
- **Input Validation:** Pydantic models prevent injection attacks
- **Error Handling:** No stack traces exposed to clients

### Scalability Considerations
- **Database:** Firestore auto-scaling
- **API:** Async endpoints for concurrent requests
- **Caching:** None implemented (can add Redis later)
- **Load Balancing:** Not implemented (server-level)
