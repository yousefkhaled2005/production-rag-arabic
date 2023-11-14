from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "Arabic Legal RAG Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "arabic_legal_corpus"
    
    DENSE_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    DEFAULT_TOP_K: int = 5
    RRF_K_CONSTANT: int = 60

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
