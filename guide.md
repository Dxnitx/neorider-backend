# NeoRider Backend Development Guide

## Running the Application

### Prerequisites
- Python 3.11+
- Firebase project with Firestore enabled
- Firebase service account key JSON file

### Setup Steps

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd neorider-backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Place your Firebase service account JSON file in the project root
   - Update `.env` with the correct path:
     ```
     FIREBASE_CREDENTIALS_PATH=./your-service-account-key.json
     ```

5. **Run the development server:**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API:**
   - API documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

### Testing
```bash
pytest
```

## ML Model Integration (Development Phase)

### Current State
The backend currently uses a **placeholder ML function** that returns hardcoded predictions. This allows development to proceed while the actual ML model is being trained.

### Placeholder Location
- **File:** `app/services/prediction_service.py`
- **Function:** `predict_behavior(features: dict) -> dict`
- **Current behavior:** Ignores input features, returns `{"prediction": "safe_riding", "confidence": 0.91, "model_used": "placeholder_v0"}`

### Feature Extraction
- **File:** `app/utils/feature_extractor.py`
- **Function:** `extract_features(sensor_window: list[dict]) -> dict`
- **Output:** 27 features (mean, std, max for 9 sensor channels)
- **Ready for:** XGBoost or scikit-learn models

### Integration Steps for Real ML Model

1. **Train your model** using the 27-feature format from `extract_features()`

2. **Replace the placeholder function:**
   ```python
   def predict_behavior(self, features: Dict[str, float]) -> Dict:
       # Load your trained model (pickle, joblib, etc.)
       model = load_your_model()

       # Prepare features as numpy array or DataFrame
       feature_vector = np.array([features[key] for key in sorted(features.keys())])

       # Make prediction
       prediction = model.predict([feature_vector])[0]
       confidence = model.predict_proba([feature_vector])[0].max()

       return {
           "prediction": prediction,
           "confidence": confidence,
           "model_used": "xgboost_v1.0"  # Update version
       }
   ```

3. **Model storage:**
   - Store trained models in `app/ml/` directory
   - Add model files to `.gitignore` if they contain sensitive data
   - Load models in the service constructor for efficiency

4. **Testing integration:**
   - Update tests in `tests/test_prediction.py` to verify real predictions
   - Ensure feature extraction produces expected input format

### Model Requirements
- **Input:** Dictionary with 27 float features
- **Output:** Dictionary with `prediction`, `confidence`, `model_used`
- **Prediction classes:** `safe_riding`, `unstable_movement`, `sudden_motion`, `risky_behavior`, `possible_accident`

### Development Workflow
1. Develop with placeholder predictions
2. Train ML model separately
3. Replace placeholder with real model
4. Test integration thoroughly
5. Update model version in responses