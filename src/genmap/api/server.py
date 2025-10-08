# src/genmap/api/server.py
import os

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from ..utils.gen_extract import extract_gen_predicates
from ..config import load_endpoints
from ..rewrite.rewriter import rewrite
from ..config import Settings
from ..index.search_candidates import search_candidates

app = FastAPI(title="genmap PoC — step6")

class TranslateIn(BaseModel):
    query: str
    endpoints_file: str = "examples/endpoints.yml"
    index_file: str = ".cache/index.json"
    use_sparse: bool = True
    use_dense: bool = True
    fusion_mode: str | None = None  # "rrf" | "wsum"; None = usa Settings
    fusion_alpha: float | None = None


class TranslateOut(BaseModel):
    mapping: dict
    rewritten: str

@app.get("/health")
def health():
    return {"status": "ok", "step": 6}

@app.post("/translate", response_model=TranslateOut)
def translate(body: TranslateIn):
    settings = Settings()

    use_sparse = getattr(body, "use_sparse", True)
    use_dense = getattr(body, "use_dense", True)
    fusion_mode = getattr(body, "fusion_mode", None) or getattr(settings, "fusion_mode", None)
    fusion_alpha = getattr(body, "fusion_alpha", None) or getattr(settings, "fusion_alpha", None)

    info = extract_gen_predicates(body.query)
    generics = info["predicates"]



    endpoints = load_endpoints(Path('./endpoints/endpoints.yml'))

    candidates = {}
    for g in generics:
        per_endpoint = search_candidates(g)
        candidates[g] = per_endpoint

    selected = {}
    for g, per_ep in candidates.items():
        selected[g] = {}
        if isinstance(per_ep, dict):
            for ep, lst in per_ep.items():
                if isinstance(lst, list) and lst:
                    if lst[0].get("score_fused") > 2:
                        pred = lst[0].get("predicate") or lst[0].get("local_name") or lst[0].get("p") or lst[0].get("uri")
                        selected[g][ep] = pred

    rewritten = rewrite(body.query, selected, endpoints)

    # subito dopo dove costruisci `selected`
    # (o filtra `gen_preds` PRIMA di usarli, se preferisci)
    def _valid_gen_key(k: str) -> bool:
        if ":" not in k:
            return False
        pref, local = k.split(":", 1)
        return bool(pref) and bool(local.strip())

    selected = {k: v for k, v in selected.items() if _valid_gen_key(k)}

    mapping = {
        "selected": selected,
        "params": {
            "use_sparse": use_sparse,
            "use_dense": use_dense,
            "dense_model": getattr(settings, "dense_model", None),
            "fusion_mode": fusion_mode,
            "fusion_alpha": fusion_alpha
        }
    }

    return TranslateOut(mapping=mapping, rewritten=rewritten)


