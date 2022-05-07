import secrets

from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V2_STR: str = "/api/v2"
    PROJECT_NAME: str = "Image Search API"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    SQLALCHEMY_DATABASE_URI: str = "postgresql://localhost:5432/waitech"

    class Config:
        env_file = ".env"


settings = Settings()