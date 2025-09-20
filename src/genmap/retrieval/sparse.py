from __future__ import annotations
from typing import Dict, List, Any, Tuple
import re, math

try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

# pesi dei campi nel documento del predicato
W_LOCAL  = 2.0
W_LABEL  = 1.6
W_EQUIV  = 1.25
W_TYPES  = 1.0   # domain/range

def _tokenize(text: str) -> List[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    toks = re.split(r"[^A-Za-z0-9]+", text)
    return [t.lower() for t in toks if len(t) >= 2]

def _doc_tokens(p) -> List[str]:
    toks = []
    toks += _tokenize(p.local_name) * int(W_LOCAL*2)
    for l in p.labels:  toks += _tokenize(l) * int(W_LABEL*2)
    for e in p.equivalents: toks += _tokenize(e.split("/")[-1]) * int(W_EQUIV*2)
    for d in p.domain: toks += _tokenize(d.split("/")[-1]) * int(W_TYPES*2)
    for r in p.range:  toks += _tokenize(r.split("/")[-1]) * int(W_TYPES*2)
    return toks or _tokenize(p.iri.split("/")[-1])

def _build_endpoint_model(endpoint_idx):
    docs, iris = [], []
    for p in endpoint_idx.predicates:
        dt = _doc_tokens(p)
        docs.append(dt); iris.append(p.iri)
    model = BM25Okapi(docs) if HAVE_BM25 and docs else None
    return {"bm25": model, "docs": docs, "iris": iris, "url": endpoint_idx.url}

def retrieve_candidates_sparse(generics: List[str], fed_index, top_k_per_endpoint:int=3) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    ep_models = {ep.id: _build_endpoint_model(ep) for ep in fed_index.endpoints}
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for g in generics:
        token = g.split(":",1)[1] if ":" in g else g
        q_toks = _tokenize(token)
        per_ep = {}
        for ep in fed_index.endpoints:
            m = ep_models[ep.id]; docs, iris = m["docs"], m["iris"]
            if not docs: per_ep[ep.id]=[]; continue
            if m["bm25"] is not None:
                scores = m["bm25"].get_scores(q_toks)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k_per_endpoint]
                per_ep[ep.id] = [{
                    "predicate": iris[i],
                    "score": float(s),
                    "reason": f"sparse: BM25(field-weighted) ~ {token}",
                    "source": "sparse"} for i,s in ranked if s>0]
            else:
                # fallback lessicale
                qset=set(q_toks)
                tmp=[]
                for i,d in enumerate(docs):
                    inter=len(qset.intersection(d))
                    if inter>0: tmp.append((i, inter/len(qset)))
                ranked=sorted(tmp,key=lambda x:x[1],reverse=True)[:top_k_per_endpoint]
                per_ep[ep.id] = [{
                    "predicate": iris[i],
                    "score": float(s),
                    "reason": f"sparse: lexical(field-weighted) ~ {token}",
                    "source": "sparse"} for i,s in ranked]
        out[g]=per_ep
    return out
