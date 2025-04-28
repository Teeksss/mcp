from pydantic import BaseModel
from typing import List, Dict

class User(BaseModel):
    username: str
    roles: List[str]
    permissions: Dict[str, List[str]]  # ör: {"model1": ["read"], "model2": ["read","write"]}