#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_indices.py
Esegue test su TUTTI gli endpoint trovati in ./indices/*/
Mostra ranking per la query su ciascun endpoint.

Uso:
  python test_indices.py -q "publish | gen:publisher" --topk 10
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import diags as spdiags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ----------------- helpers -----------------

def _tokenize_label(s: str) -> str:
    if not s:
        return ""
    seg = s.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    seg = re.sub(r"[_\\-\\.\\+]+", " ", seg)
    seg = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", seg)
    try:
        from urllib.parse import unquote
        seg = unquote(seg)
    except Exception:
        pass
    return seg.lower()

def _compose_query_text(q: str) -> str:
    return " ".join([q, _tokenize_label(q)])

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.nanmean(x)); sd = float(np.nanstd(x)) + 1e-9
    return (x - mu) / sd

def _load_meta(dirpath: Path) -> pd.DataFrame:
    p_par = dirpath / "meta.parquet"
    p_csv = dirpath / "meta.csv"
    if p_par.exists():
        return pd.read_parquet(p_par)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"meta.parquet/meta.csv non trovati in {dirpath}")

def _load_sparse(dirpath: Path) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    sp_path = dirpath / "sparse.npz"
    voc_path = dirpath / "vocab.json"
    if not sp_path.exists() or not voc_path.exists():
        raise FileNotFoundError(f"sparse.npz o vocab.json mancanti in {dirpath}")
    X = sparse.load_npz(sp_path)
    vocab = json.loads(voc_path.read_text(encoding="utf-8"))

    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        strip_accents="unicode", lowercase=True, dtype=np.float32,
        vocabulary=vocab
    )
    # carico idf.json se esiste
    idf_path = dirpath / "idf.json"
    if idf_path.exists():
        idf = np.array(json.loads(idf_path.read_text(encoding="utf-8")), dtype=np.float64)
    else:
        idf = np.ones(len(vocab), dtype=np.float64)

    n_feat = len(vocab)
    if idf.size != n_feat:
        fix = np.ones(n_feat, dtype=np.float64)
        m = min(n_feat, idf.size)
        fix[:m] = idf[:m]
        idf = fix

    vec.idf_ = idf
    vec._tfidf._idf_diag = spdiags(idf, offsets=0, shape=(n_feat, n_feat), dtype=np.float64)
    return vec, X

def _maybe_load_dense(dirpath: Path):
    dense_path = dirpath / "dense.npy"
    model_path = dirpath / "dense_model.txt"
    if not dense_path.exists() or not model_path.exists():
        return None, None, None
    D = np.load(dense_path)
    model_name = model_path.read_text(encoding="utf-8").strip()
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)
    except Exception:
        st = None
    return D, model_name, st

def _pretty_table(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    headers = ["rank", "endpoint", "local_name", "score_fused", "score_sparse", "score_dense", "usage"]
    lines = []
    lines.append(" | ".join(h.ljust(14) for h in headers))
    lines.append("-+-".join("-"*14 for _ in headers))
    for i, r in enumerate(rows[:max_rows], 1):
        lines.append(" | ".join([
            str(i).ljust(14),
            str(r.get("endpoint","")).ljust(14),
            str(r.get("local_name",""))[:40].ljust(14),
            f"{r.get('score_fused',0):.4f}".ljust(14),
            f"{r.get('score_sparse',0):.4f}".ljust(14),
            f"{r.get('score_dense',0):.4f}".ljust(14),
            str(r.get("usage_count",0)).ljust(14),
        ]))
    return "\n".join(lines)


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", nargs="+", required=True, help="Una o più query di test")
    ap.add_argument("--indices-root", default="./indices", help="radice indici")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--w_sparse", type=float, default=0.7)
    ap.add_argument("--w_dense", type=float, default=0.3)
    ap.add_argument("--popa", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    indices_root = Path(args.indices_root)
    endpoints = [p for p in indices_root.iterdir() if p.is_dir()]

    for ep_dir in endpoints:
        endpoint = ep_dir.name
        print(f"\n=== Endpoint: {endpoint} ===")

        try:
            meta = _load_meta(ep_dir)
            vec, Xs = _load_sparse(ep_dir)
            dense_triplet = _maybe_load_dense(ep_dir)
        except Exception as e:
            print(f"[warn] skip {endpoint}: {e}")
            continue

        for q in args.query:
            print(f"\n--- QUERY: {q} ---")
            qtext = _compose_query_text(q)
            Xq = vec.transform([qtext])

            Xs_n = normalize(Xs, norm="l2", copy=False)
            Xq_n = normalize(Xq, norm="l2", copy=False)
            score_sparse = (Xq_n @ Xs_n.T).toarray().ravel().astype(np.float32)

            score_dense = np.zeros(len(meta), dtype=np.float32)
            if dense_triplet is not None:
                D, model_name, st = dense_triplet
                if D is not None and st is not None:
                    q_emb = st.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
                    score_dense = (D @ q_emb).astype(np.float32)

            s = args.w_sparse * _zscore(score_sparse) + args.w_dense * _zscore(score_dense)

            if "usage_count" in meta.columns and args.popa != 0.0:
                pop = np.log1p(meta["usage_count"].fillna(0).to_numpy(np.float32))
                s += args.popa * _zscore(pop)

            k = int(min(args.topk, len(meta)))
            idx = np.argpartition(-s, k-1)[:k]
            idx = idx[np.argsort(-s[idx])]

            results = []
            for i in idx:
                row = meta.iloc[i]
                results.append({
                    "endpoint": row.get("endpoint"),
                    "predicate": row.get("predicate"),
                    "local_name": row.get("local_name"),
                    "usage_count": int(row.get("usage_count") or 0),
                    "score_sparse": float(score_sparse[i]),
                    "score_dense": float(score_dense[i]),
                    "score_fused": float(s[i]),
                })

            print(_pretty_table(results, max_rows=args.topk))
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
