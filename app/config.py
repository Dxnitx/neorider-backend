from pydantic import BaseSettings
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    firebase_credentials_path: str
    app_env: str = "development"
    app_version: str = "1.0.0"

    model_config = {"env_file": ".env"}

@lru_cache()
def get_settings() -> Settings:
    return Settings()