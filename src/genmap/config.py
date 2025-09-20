# src/genmap/config.py
from pathlib import Path
from pydantic import BaseModel
import yaml
import os

class Settings(BaseModel):
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "local-json"
    llm_timeout_s: int = 60
    top_k_per_endpoint: int = 3
    index_path: Path = Path(".cache/index.sqlite")
    cache_path: Path = Path(".cache/mapping.sqlite")
    openai_model: str = os.getenv("GENMAP_OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    use_openai: bool = True
    top_k_per_endpoint: int = 3
    use_sparse: bool = True
    use_dense: bool = True
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fusion_mode: str = "rrf"   # "rrf" | "wsum"
    fusion_alpha: float = 0.6

def load_endpoints(path: Path):
    data = yaml.safe_load(Path(path).read_text())
    return data["endpoints"]
