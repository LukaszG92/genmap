# src/genmap/api/server.py
import os
import pprint

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from ..utils.gen_extract import extract_gen_predicates
from ..config import load_endpoints
from ..rewrite.rewriter import rewrite
from ..config import Settings
from ..index.search_candidates import search_candidates
from ..llm.openai_client import one_shot_map_openai

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

    selected = one_shot_map_openai("gpt-5-nano", body.query, generics, candidates)

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
        "candidates": candidates,
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