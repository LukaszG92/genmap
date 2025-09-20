# src/genmap/api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from ..utils.gen_extract import extract_gen_predicates
from ..config import load_endpoints
from ..rewrite.rewriter import rewrite
from ..index.loaders import load_index
from ..retrieval.sparse import retrieve_candidates_sparse
from ..retrieval.candidates_mock import plan_mock_candidates
from ..config import Settings
from ..llm.openai_client import one_shot_map_openai, OpenAIError
from ..llm.selector import selected_from_llm

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
    # 0) config + input
    settings = Settings()

    # 1) estrai i gen:* dalla query
    info = extract_gen_predicates(body.query)
    generics = info["predicates"]

    # 2) carica endpoints e indice
    endpoints = load_endpoints(Path(body.endpoints_file))
    idx = load_index(Path(body.index_file))

    # 3) retrieval ibrido (sparse + dense + fusione)
    top_k = settings.top_k_per_endpoint
    use_sparse = body.use_sparse if body.use_sparse is not None else settings.use_sparse
    use_dense  = body.use_dense  if body.use_dense  is not None else settings.use_dense
    fusion_mode = body.fusion_mode or settings.fusion_mode
    fusion_alpha = body.fusion_alpha if body.fusion_alpha is not None else settings.fusion_alpha

    candidates = retrieve_candidates_hybrid(
        generics=generics,
        fed_index=idx,
        top_k_per_endpoint=top_k,
        use_sparse=use_sparse,
        use_dense=use_dense,
        dense_model=settings.dense_model,
        fusion_mode=fusion_mode,
        alpha=fusion_alpha
    )

    # 4) riscrittura SPARQL (usa i candidati "effective")
    rewritten = rewrite(body.query, candidates, endpoints)

    # 5) payload con un po' di diagnostica utile
    mapping = {
        "found_generics": generics,
        "sample_triples": info["triples"],
        "candidates": candidates,
        "retrieval": {
            "index_version": idx.version,
            "top_k_per_endpoint": top_k,
            "use_sparse": use_sparse,
            "use_dense": use_dense,
            "dense_model": settings.dense_model,
            "fusion_mode": fusion_mode,
            "fusion_alpha": fusion_alpha
        }
    }
    return TranslateOut(mapping=mapping, rewritten=rewritten)



# in cima dove hai gli import


# in fondo al file, aggiungi questa rotta:
@app.get("/debug/index")
def debug_index(path: str = ".cache/index.json"):
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"index not found at {p}"}
    idx = load_index(p)
    summary = [
        {
            "id": e.id,
            "url": e.url,
            "predicates": len(e.predicates),
            "sampled": e.triples_sampled
        }
        for e in idx.endpoints
    ]
    return {"ok": True, "version": idx.version, "endpoints": summary}\

@app.get("/debug/retrieve")
def debug_retrieve(term: str, index_file: str = ".cache/index.json", k: int = 5):
    idx = load_index(Path(index_file))
    cands = retrieve_candidates_sparse([f"gen:{term}"], idx, top_k_per_endpoint=k)
    return {"term": term, "candidates": cands.get(f"gen:{term}", {})}

@app.post("/debug/llm")
def debug_llm(body: TranslateIn):
    from ..index.loaders import load_index
    from ..retrieval.sparse import retrieve_candidates_sparse
    settings = Settings()

    info = extract_gen_predicates(body.query)
    generics = info["predicates"]
    endpoints = load_endpoints(Path(body.endpoints_file))

    idx = load_index(Path(body.index_file))
    base_cands = retrieve_candidates_sparse(generics, idx, top_k_per_endpoint=3)

    out = {"llm_used": False, "ok": False, "error": None, "raw": None}
    if settings.use_openai:
        try:
            raw = one_shot_map_openai(
                model=settings.openai_model,
                query=body.query,
                generics=generics,
                candidates=base_cands,
                timeout_s=settings.llm_timeout_s
            )
            out.update(llm_used=True, ok=True, raw=raw)
        except OpenAIError as e:
            out.update(llm_used=True, ok=False, error=str(e))
    return out

