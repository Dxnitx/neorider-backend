import firebase_admin
from firebase_admin import credentials, firestore
from app.config import get_settings

def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials."""
    settings = get_settings()
    cred = credentials.Certificate(settings.firebase_credentials_path)
    firebase_admin.initialize_app(cred)

def get_firestore_client():
    """Get Firestore client instance."""
    return firestore.client()