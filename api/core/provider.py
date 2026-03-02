from api.core.settings import settings
from inference.model_client import MockModelClient, VLLMModelClient, OpenAIModelClient

def get_model_client():

    # Fetch provider from env, otherwise default to mock
    provider = settings.MODEL_PROVIDER.lower()

    if provider == "mock":
        return MockModelClient()
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        return OpenAIModelClient(api_key=settings.OPENAI_API_KEY)
    elif provider == "vllm":
        return VLLMModelClient()
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")