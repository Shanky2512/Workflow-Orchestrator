import requests
from typing import List, Dict

class LLMClientDetails:
    """
    Core LLM Inspector
    No Fast API, No globals, Pure logic
    """

    def __init__(self, base_url:str, timeout:int = 30):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout

    def fetch_models(self) -> List[Dict]:
        url = f"{self.base_url}/api/tags"
        resp = requests.get(url,timeout=self.timeout)
        resp.raise_for_status()
        
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid LLM resposne: expected dict")
        return payload.get("models",[])
    
    def fetch_model_names(self) -> List[Dict]:
        models = self.fetch_models()
        return [m["name"] for m in models if "name" in m and not m["name"].startswith("nomic-embed")]
    
    def fetch_llm_name(self) -> str:
        return "ollama"
    
    