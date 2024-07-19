import os

class Config:
    ELASTICSEARCH_HOSTS = os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "dataset_name")
