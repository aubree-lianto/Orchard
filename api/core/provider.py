from app.core.settings import settings
from inference.model_client import MockModelClient, VLLMModelClient

def get_model_client():

    # Fetch provider from env, otherwise default to mock
    provider = settings.MODEL_PROVIDER.lower()

    if provider == "mock":
        return MockModelClient()
    elif provider ==  "vllm":
        return VLLMModelClient()
    else: 
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")