# src/genmap/index/loaders.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, Optional

from .schema import FederationIndex, EndpointIndex, PredicateInfo

def load_index(path: Path) -> FederationIndex:
    data = json.loads(Path(path).read_text())
    # ricostruisci i modelli Pydantic
    eps = []
    for e in data["endpoints"]:
        preds = [PredicateInfo(**p) for p in e.get("predicates", [])]
        eps.append(EndpointIndex(id=e["id"], url=e["url"], built_at=e.get("built_at", 0),
                                 predicates=preds, triples_sampled=e.get("triples_sampled")))
    return FederationIndex(version=data["version"], endpoints=eps)

def endpoint_map(idx: FederationIndex) -> Dict[str, EndpointIndex]:
    return {e.id: e for e in idx.endpoints}
