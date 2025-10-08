import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- utils ----------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _tokenize_label(s: str) -> str:
    if not s:
        return ""
    seg = s.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    seg = re.sub(r"[_\-\.\+]+", " ", seg)
    seg = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", seg)
    try:
        from urllib.parse import unquote
        seg = unquote(seg)
    except Exception:
        pass
    return seg.lower()

def _compose_text(endpoint: str, predicate: str, local_name: str) -> str:
    ep_short = endpoint.rsplit("/", 1)[-1] if endpoint else ""
    toks = [
        predicate or "",
        local_name or "",
        _tokenize_label(predicate or ""),
        _tokenize_label(local_name or ""),
        _tokenize_label(ep_short),
    ]
    return " ".join([t for t in toks if t])

def _build_sparse(texts: List[str], max_features: int = 200_000):
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        min_df=1, max_df=1.0, max_features=max_features,
        strip_accents="unicode", lowercase=True, dtype=np.float32,
    )
    X = vec.fit_transform(texts)
    return vec, X.tocsr()

def _build_dense(texts: List[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )
    return embs.astype(np.float32)

# ---------------- normalizzazione input ----------------

def _normalize_single_endpoint_payload(obj: Dict[str, Any], endpoint: str) -> List[Dict[str, Any]]:
    """
    Restituisce lista di record per UN solo endpoint.
    Supporta:
      A) { "endpoints": [ { "id": endpoint, "predicates": [...] } ] }
      B) { "id": endpoint, "predicates": [...] }
      C) { "predicates": [...] }  # endpoint ricavato dal parametro CLI
    """
    # Caso A
    eps = obj.get("endpoints")
    if isinstance(eps, list) and len(eps) > 0:
        # se c'è un match sul parametro, prendi quello; altrimenti il primo
        chosen = None
        for ep in eps:
            if isinstance(ep, dict) and (ep.get("id") == endpoint):
                chosen = ep
                break
        if chosen is None:
            chosen = eps[0]
        preds = chosen.get("predicates") or []
        return _records_from_predicates(preds, endpoint=endpoint or chosen.get("id") or "")
    # Caso B
    if isinstance(obj.get("predicates"), list) and (obj.get("id") is not None):
        return _records_from_predicates(obj["predicates"], endpoint=endpoint or obj.get("id") or "")
    # Caso C
    if isinstance(obj.get("predicates"), list):
        return _records_from_predicates(obj["predicates"], endpoint=endpoint or "")
    raise ValueError("Struttura JSON non riconosciuta per singolo endpoint.")

def _records_from_predicates(preds: List[Any], endpoint: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in preds:
        if not isinstance(p, dict):
            # se fosse stringa IRI (caso raro)
            iri = str(p)
            local = iri.rsplit("/", 1)[-1]
            cnt = 0
        else:
            iri = str(p.get("iri") or "").strip()
            if not iri:
                continue
            local = str(p.get("local_name") or iri.rsplit("/", 1)[-1])
            try:
                cnt = int(p.get("usage_count") or 0)
            except Exception:
                cnt = 0
        out.append({
            "endpoint": endpoint,
            "predicate": iri,
            "local_name": local,
            "usage_count": cnt,
        })
    return out


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Costruisce indici per un singolo endpoint.")
    ap.add_argument("-e", "--endpoint", required=True, help="id endpoint (es. affymetrix)")
    ap.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dumps-root", default="./dumps", help="radice degli input (default: ./dumps)")
    ap.add_argument("--indices-root", default="./indices", help="radice degli output (default: ./indices)")
    args = ap.parse_args()

    in_path = Path(args.dumps_root) / "out" / args.endpoint / "predicates.json"
    out_dir = Path(args.indices_root) / args.endpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = _read_json(in_path)
    rows = _normalize_single_endpoint_payload(obj, endpoint=args.endpoint)
    if not rows:
        raise ValueError(f"Nessun predicato trovato in {in_path}")

    # testo + dataframe
    for r in rows:
        r["text"] = _compose_text(r["endpoint"], r["predicate"], r["local_name"])
    df = pd.DataFrame(rows, columns=["endpoint", "predicate", "local_name", "usage_count", "text"])

    # Sparse
    print(f"[sparse] TF-IDF on {len(df)} rows ...")
    vec, X = _build_sparse(df["text"].tolist(), max_features=args.max_features)

    # Salvataggi base
    sparse.save_npz(out_dir / "sparse.npz", X)

    # vocab.json (cast a tipi JSON-compatibili)
    vocab_items = sorted(vec.vocabulary_.items(), key=lambda kv: int(kv[1]))
    vocab_json = {str(k): int(v) for k, v in vocab_items}
    (out_dir / "vocab.json").write_text(json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8")

    # idf.json per ricostruire correttamente il vectorizer a runtime
    idf = getattr(vec, "idf_", None)
    if idf is not None:
        (out_dir / "idf.json").write_text(json.dumps([float(x) for x in idf]), encoding="utf-8")

    # meta
    try:
        df.to_parquet(out_dir / "meta.parquet", index=False)
    except Exception as e:
        print(f"[warn] parquet non disponibile ({e}); salvo meta.csv")
        df.to_csv(out_dir / "meta.csv", index=False)

    # Dense (opzionale)
    if args.dense_model:
        print(f"[dense] encoding with model={args.dense_model} ...")
        embs = _build_dense(df["text"].tolist(), model_name=args.dense_model, batch_size=args.batch_size)
        np.save(out_dir / "dense.npy", embs)
        (out_dir / "dense_model.txt").write_text(args.dense_model + "\n", encoding="utf-8")
    else:
        print("[dense] skipped")

    print("[done] wrote indices to", str(out_dir))


if __name__ == "__main__":
    main()
