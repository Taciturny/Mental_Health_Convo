# from pydantic_settings import BaseSettings
from pydantic import BaseSettings


class Settings(BaseSettings):
    COHERE_API_KEY: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    DEPLOYMENT_MODE: str = 'local'  # 'local' or 'cloud'
    MODEL_TYPE: str = 'local'  # 'local' or 'cohere'

    class Config:
        env_file = ".env"

settings = Settings()
