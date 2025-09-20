from __future__ import annotations
from typing import Dict, List, Any
from collections import defaultdict

def _normalize(scores: List[float]) -> List[float]:
    if not scores: return scores
    mx=max(scores); mn=min(scores)
    if mx==mn: return [1.0 for _ in scores]
    return [(s-mn)/(mx-mn) for s in scores]

def fuse_endpoint_lists(sparse: List[Dict[str,Any]],
                        dense: List[Dict[str,Any]],
                        mode: str = "rrf",
                        alpha: float = 0.6,
                        rrf_k: int = 60) -> List[Dict[str,Any]]:
    """
    Unisce due liste di candidati (stessa endpoint) per 'predicate' IRI.
    mode:
      - "rrf": Reciprocal Rank Fusion (somma 1/(k+rank))
      - "wsum": alpha*norm(sparse) + (1-alpha)*norm(dense)
    """
    # indicizza per IRI
    by_pred = defaultdict(lambda: {"sources": {}, "reason_parts": []})
    # sparse
    s_sorted = sorted(sparse, key=lambda x: x.get("score",0), reverse=True)
    for rank, c in enumerate(s_sorted, start=1):
        iri=c["predicate"]; by_pred[iri]["sources"]["sparse"]=(c["score"], rank); by_pred[iri]["reason_parts"].append(c["reason"])
    # dense
    d_sorted = sorted(dense, key=lambda x: x.get("score",0), reverse=True)
    for rank, c in enumerate(d_sorted, start=1):
        iri=c["predicate"]; by_pred[iri]["sources"]["dense"]=(c["score"], rank); by_pred[iri]["reason_parts"].append(c["reason"])

    out=[]
    if mode=="rrf":
        for iri,info in by_pred.items():
            s_rank = info["sources"].get("sparse",(0.0, None))[1]
            d_rank = info["sources"].get("dense",(0.0, None))[1]
            score = 0.0
            if s_rank is not None: score += 1.0 / (rrf_k + s_rank)
            if d_rank is not None: score += 1.0 / (rrf_k + d_rank)
            out.append({"predicate": iri, "score": score, "reason": "fusion: RRF(sparse,dense)", "source": "fusion"})
    else:  # weighted sum
        s_vals=[c["score"] for c in s_sorted]; d_vals=[c["score"] for c in d_sorted]
        s_norm=_normalize(s_vals); d_norm=_normalize(d_vals)
        s_map={c["predicate"]: s_norm[i] for i,c in enumerate(s_sorted)}
        d_map={c["predicate"]: d_norm[i] for i,c in enumerate(d_sorted)}
        for iri,info in by_pred.items():
            sv=s_map.get(iri,0.0); dv=d_map.get(iri,0.0)
            score = alpha*sv + (1.0-alpha)*dv
            out.append({"predicate": iri, "score": score, "reason": f"fusion: wsum(alpha={alpha})", "source": "fusion"})
    # ordina per score desc
    out.sort(key=lambda x:x["score"], reverse=True)
    return out
