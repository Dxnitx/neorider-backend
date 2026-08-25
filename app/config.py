from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 🔥 Firebase (REQUIRED)
    firebase_credentials_path: str

    # Optional
    app_env: str = "development"
    app_version: str = "1.0.0"

    model_config = {
        "env_file": ".env",
        "extra": "allow"   # optional safety (can ignore extra fields)
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()