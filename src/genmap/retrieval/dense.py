from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np

def _has_st():
    try:
        import sentence_transformers  # type: ignore
        return True
    except Exception:
        return False

def _embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

def _predicate_text(p) -> str:
    # testo denso: local_name + labels + equivalents (localnames) + domain/range (localnames)
    pieces = [p.local_name] + p.labels
    pieces += [e.split("/")[-1] for e in p.equivalents]
    pieces += [d.split("/")[-1] for d in p.domain] + [r.split("/")[-1] for r in p.range]
    return " ".join(pieces) or p.iri

def build_dense_models(fed_index, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Costruisce per endpoint: vettori (N x D) e lista IRIs. Fallback: None se ST non disponibile."""
    if not _has_st():  # fallback: dense disabilitato
        return {ep.id: {"embs": None, "iris": [], "url": ep.url} for ep in fed_index.endpoints}
    out={}
    for ep in fed_index.endpoints:
        texts=[_predicate_text(p) for p in ep.predicates]
        if not texts:
            out[ep.id]={"embs": None,"iris": [],"url":ep.url}; continue
        embs=_embed_texts(model_name, texts)
        iris=[p.iri for p in ep.predicates]
        out[ep.id]={"embs": embs, "iris": iris, "url": ep.url}
    return out

def retrieve_candidates_dense(generics: List[str], fed_index, dense_models, top_k_per_endpoint:int=3) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out={}
    if not generics: return {}
    # costruiamo query embedding per 'birthDate'→"birth date"
    def norm_query(g):
        tok=g.split(":",1)[1] if ":" in g else g
        return " ".join(np.char.split(tok, sep=r"(?=[A-Z])")) if any(c.isupper() for c in tok) else tok
    if dense_models and any(dense_models[ep.id]["embs"] is not None for ep in fed_index.endpoints):
        # prepara embedder una volta sola
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_texts=[norm_query(g) for g in generics]
        Q=model.encode(q_texts, show_progress_bar=False, normalize_embeddings=True)
    else:
        # niente dense
        return {g:{ep.id:[] for ep in fed_index.endpoints} for g in generics}

    for gi,g in enumerate(generics):
        per_ep={}
        q=Q[gi]
        for ep in fed_index.endpoints:
            m=dense_models[ep.id]
            if m["embs"] is None or len(m["iris"])==0:
                per_ep[ep.id]=[]; continue
            embs=m["embs"]; iris=m["iris"]
            # cosine già normalizzata = dot product
            sims = embs @ q
            top_idx = np.argsort(-sims)[:top_k_per_endpoint]
            per_ep[ep.id]=[{
                "predicate": iris[i],
                "score": float(sims[i]),
                "reason": "dense: cosine on ST embeddings",
                "source": "dense"
            } for i in top_idx if sims[i] > 0]
        out[g]=per_ep
    return out
