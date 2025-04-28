from typing import List, Dict, Optional
import random

class EndpointUnavailable(Exception):
    pass

class SmartRouter:
    def __init__(self, endpoints: Dict[str, List[Dict]]):
        self.endpoints = endpoints

    def get_healthy_endpoints(self, model_name: str) -> List[Dict]:
        candidates = self.endpoints.get(model_name, [])
        return [ep for ep in candidates if ep.get("health", False)]

    def select_lowest_load(self, endpoints: List[Dict]) -> Optional[Dict]:
        if not endpoints:
            return None
        return min(endpoints, key=lambda ep: ep.get("load", 1.0))

    def route(self, model_name: str) -> str:
        healthy_eps = self.get_healthy_endpoints(model_name)
        if not healthy_eps:
            raise EndpointUnavailable(f"No healthy endpoint for model: {model_name}")
        lowest_load = self.select_lowest_load(healthy_eps)
        candidates = [ep for ep in healthy_eps if ep["load"] == lowest_load["load"]]
        chosen = random.choice(candidates)
        return chosen["url"]