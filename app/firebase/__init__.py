from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from app.config import get_settings


def _resolve_credentials_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def initialize_firebase() -> None:
    settings = get_settings()
    credentials_path = _resolve_credentials_path(settings.firebase_credentials_path)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Firebase credentials file not found: {credentials_path}. "
            "Set FIREBASE_CREDENTIALS_PATH in .env to a valid path."
        )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(credentials_path)))


def get_firestore_client():
    initialize_firebase()
    return firestore.client()