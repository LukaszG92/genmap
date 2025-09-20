from __future__ import annotations
from typing import Dict, List, Any
from .sparse import retrieve_candidates_sparse
from .dense import build_dense_models, retrieve_candidates_dense
from .fusion import fuse_endpoint_lists

def retrieve_candidates_hybrid(generics: List[str], fed_index,
                               top_k_per_endpoint:int=3,
                               use_sparse: bool = True,
                               use_dense: bool = True,
                               dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                               fusion_mode: str = "rrf",
                               alpha: float = 0.6) -> Dict[str, Dict[str, List[Dict[str,Any]]]]:
    """
    Per ogni gen:* e per endpoint, fonde sparse+dense. Se uno dei due manca, usa l'altro.
    """
    out={}
    # sparse
    S = retrieve_candidates_sparse(generics, fed_index, top_k_per_endpoint) if use_sparse else {g:{} for g in generics}
    # dense
    D = {}
    if use_dense:
        dense_models = build_dense_models(fed_index, model_name=dense_model)
        D = retrieve_candidates_dense(generics, fed_index, dense_models, top_k_per_endpoint)
    else:
        D = {g:{ep.id:[] for ep in fed_index.endpoints} for g in generics}

    for g in generics:
        per_ep={}
        for ep in fed_index.endpoints:
            s = (S.get(g,{})).get(ep.id, [])
            d = (D.get(g,{})).get(ep.id, [])
            if s and d:
                fused = fuse_endpoint_lists(s, d, mode=fusion_mode, alpha=alpha)
                per_ep[ep.id]=fused[:top_k_per_endpoint]
            elif s:
                per_ep[ep.id]=s
            else:
                per_ep[ep.id]=d
        out[g]=per_ep
    return out
