from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    COHERE_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    DEPLOYMENT_MODE: str = (
        "local"  # Default to 'cloud' for Cohere and Qdrant cloud (when running deployment/main.py)
    )
    COLLECTION_NAME_CLOUD: str
    COLLECTION_NAME_LOCAL: str
    DATA_FILE_PATH: str
    GROUND_TRUTH_DATA: str

    # Database configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    MONITORING_DB: str

    # Grafana configuration
    GF_SECURITY_ADMIN_USER: str
    GF_SECURITY_ADMIN_PASSWORD: str

    # LLM model parameters
    MAX_NEW_TOKENS: int = 50
    NUM_RETURN_SEQUENCES: int = 1
    TEMPERATURE: float = 0.8
    TOP_K: int = 50
    TOP_P: float = 0.95
    REPETITION_PENALTY: float = 1.0
    DO_SAMPLE: bool = True

    # Model names
    GPT2_MODEL: str = "gpt2-medium"
    DIALOGPT_MODEL: str = "microsoft/DialoGPT-medium"
    DISTILGPT2_MODEL: str = "distilgpt2"
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

    class Config:
        # env_file = ".env"
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        # env_file = str(Path(__file__).resolve().parent.parent / '.env') uncomment when running locally
        # env_file_encoding = "utf-8"


settings = Settings()
