from typing import Dict
from src.services.intelligence.llm_client import LLMClient

class ModelManager:
    def __init__(self):
        self.models: Dict[str, LLMClient] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("phi3", provider="transformers", model_name="microsoft/phi-3-mini-4k-instruct", device="cpu")
        # self.register("gpt4", provider="openai", model_name="gpt-4", api_key=...)

    def register(self, key: str, provider: str, model_name: str, api_key: str = None, device: str = "cpu", **kwargs):
        self.models[key] = LLMClient(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            device=device,
            **kwargs
        )

    def get(self, key: str) -> LLMClient:
        if key not in self.models:
            raise ValueError(f"Model {key} kayıtlı değil.")
        return self.models[key]

    def generate(self, key: str, prompt: str, **kwargs) -> str:
        client = self.get(key)
        return client.generate(prompt, **kwargs)

mm = ModelManager()