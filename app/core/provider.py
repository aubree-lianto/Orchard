import os
from inference.model_client import MockModelClient, VLLMModelClient

def get_model_client():

    # Fetch provider from env, otherwise default to mock
    provider = os.getenv("MODEL_PROVIDER", "mock").lower()

    if provider == "mock":
        return MockModelClient()
    elif provider ==  "vllm":
        return VLLMModelClient()
    else: 
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")