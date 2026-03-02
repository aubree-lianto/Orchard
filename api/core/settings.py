from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # If variable missing from .env fall back to defaults
    MODEL_PROVIDER: str = "mock"
    MODEL_BASE_URL: str = "http://localhost:8000"

    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # LLM API keys
    OPENAI_API_KEY: str = ""

    # Tool API keys
    TAVILY_API_KEY: str = ""
    ARXIV_MAX_RESULTS: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Global settings instance
settings = Settings()