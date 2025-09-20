from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import hashlib, json, time

class PredicateInfo(BaseModel):
    iri: str
    local_name: str
    usage_count: Optional[int] = None
    labels: List[str] = []          # ⬅️ NEW
    domain: List[str] = []          # ⬅️ NEW
    range: List[str] = []           # ⬅️ NEW
    equivalents: List[str] = []     # ⬅️ NEW

class EndpointIndex(BaseModel):
    id: str
    url: str
    built_at: float = Field(default_factory=lambda: time.time())
    predicates: List[PredicateInfo] = []
    triples_sampled: Optional[int] = None

class FederationIndex(BaseModel):
    version: str
    endpoints: List[EndpointIndex]

def compute_version(payload: Dict) -> str:
    frozen = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(frozen.encode("utf-8")).hexdigest()[:16]
