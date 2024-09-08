# from pydantic_settings import BaseSettings
# from dotenv import load_dotenv
# from pathlib import Path
# import os

# # Load .env file if it exists (for local development)
# env_path = Path(__file__).resolve().parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)

import os
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    DEPLOYMENT_MODE: str = os.environ.get("DEPLOYMENT_MODE", "cloud")  # Default to 'cloud' for Render
    COLLECTION_NAME_CLOUD: str = os.environ.get("COLLECTION_NAME_CLOUD", "")
    COLLECTION_NAME_LOCAL: str = os.environ.get("COLLECTION_NAME_LOCAL", "")
    DATA_FILE_PATH: str = os.environ.get("DATA_FILE_PATH", "")
    GROUND_TRUTH_DATA: str = os.environ.get("GROUND_TRUTH_DATA", "")

    # Database configuration
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
    POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "")
    MONITORING_DB: str = os.environ.get("MONITORING_DB", "")

    # Grafana configuration
    GF_SECURITY_ADMIN_USER: str = os.environ.get("GF_SECURITY_ADMIN_USER", "")
    GF_SECURITY_ADMIN_PASSWORD: str = os.environ.get("GF_SECURITY_ADMIN_PASSWORD", "")

    # LLM model parameters
    MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", 50))
    NUM_RETURN_SEQUENCES: int = int(os.environ.get("NUM_RETURN_SEQUENCES", 1))
    TEMPERATURE: float = float(os.environ.get("TEMPERATURE", 0.8))
    TOP_K: int = int(os.environ.get("TOP_K", 50))
    TOP_P: float = float(os.environ.get("TOP_P", 0.95))
    REPETITION_PENALTY: float = float(os.environ.get("REPETITION_PENALTY", 1.0))
    DO_SAMPLE: bool = os.environ.get("DO_SAMPLE", "True").lower() == "true"

    # Model names
    GPT2_MODEL: str = os.environ.get("GPT2_MODEL", 'gpt2-medium')
    DIALOGPT_MODEL: str = os.environ.get("DIALOGPT_MODEL", 'microsoft/DialoGPT-medium')
    DISTILGPT2_MODEL: str = os.environ.get("DISTILGPT2_MODEL", 'distilgpt2')
    SENTIMENT_MODEL: str = os.environ.get("SENTIMENT_MODEL", 'distilbert-base-uncased-finetuned-sst-2-english')

    class Config:
        env_file = ".env"
        # env_file = str(Path(__file__).resolve().parent.parent / '.env') uncomment when running locally
        env_file_encoding = "utf-8"

settings = Settings()

