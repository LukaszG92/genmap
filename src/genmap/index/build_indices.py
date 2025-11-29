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

def process_endpoint(endpoint: str, dumps_root: Path, indices_root: Path, max_features: int) -> bool:
    """
    Processa un singolo endpoint. Ritorna True se processato, False se saltato.
    """
    in_path = dumps_root / endpoint / "predicates.json"
    out_dir = indices_root / endpoint

    # Controlla se l'indice esiste già
    if (out_dir / "sparse.npz").exists() and (out_dir / "vocab.json").exists():
        print(f"[{endpoint}] ⊘ Indici già esistenti, salto")
        return False

    if not in_path.exists():
        print(f"[{endpoint}] ⚠️  File {in_path} non trovato, salto")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    obj = _read_json(in_path)
    rows = _normalize_single_endpoint_payload(obj, endpoint=endpoint)
    if not rows:
        print(f"[{endpoint}] ⚠️  Nessun predicato trovato, salto")
        return False

    # testo + dataframe
    for r in rows:
        r["text"] = _compose_text(r["endpoint"], r["predicate"], r["local_name"])
    df = pd.DataFrame(rows, columns=["endpoint", "predicate", "local_name", "usage_count", "text"])

    # Sparse
    print(f"[{endpoint}] Costruzione indice TF-IDF su {len(df)} predicati...")
    vec, X = _build_sparse(df["text"].tolist(), max_features=max_features)

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
        print(f"[{endpoint}] [warn] parquet non disponibile ({e}); salvo meta.csv")
        df.to_csv(out_dir / "meta.csv", index=False)

    print(f"[{endpoint}] ✓ Indici salvati in {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Costruisce indici TF-IDF per endpoint. Se --endpoint non specificato, processa tutti gli endpoint trovati."
    )
    ap.add_argument("-e", "--endpoint", help="id endpoint specifico (es. affymetrix). Se omesso, processa tutti.")
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--dumps-root", default="./predicates", help="radice degli input (default: ./predicates)")
    ap.add_argument("--indices-root", default="./indices", help="radice degli output (default: ./indices)")
    args = ap.parse_args()

    dumps_root = Path(args.dumps_root)
    indices_root = Path(args.indices_root)

    if args.endpoint:
        # Processa singolo endpoint
        success = process_endpoint(args.endpoint, dumps_root, indices_root, args.max_features)
        if not success:
            print(f"\nNessun indice creato per {args.endpoint}")
    else:
        # Processa tutti gli endpoint trovati
        if not dumps_root.exists():
            raise ValueError(f"Directory {dumps_root} non trovata")

        # Trova tutti gli endpoint (sottocartelle con predicates.json)
        endpoints = []
        for item in dumps_root.iterdir():
            if item.is_dir() and (item / "predicates.json").exists():
                endpoints.append(item.name)

        if not endpoints:
            print(f"Nessun endpoint trovato in {dumps_root}")
            return

        print(f"Trovati {len(endpoints)} endpoint da processare\n")

        processed = 0
        skipped = 0

        for idx, endpoint in enumerate(sorted(endpoints), 1):
            print(f"[{idx}/{len(endpoints)}] Processamento {endpoint}...")
            success = process_endpoint(endpoint, dumps_root, indices_root, args.max_features)
            if success:
                processed += 1
            else:
                skipped += 1
            print()

        print(f"Completato:")
        print(f"  - Processati: {processed}")
        print(f"  - Saltati: {skipped}")
        print(f"  - Totale: {len(endpoints)}")


if __name__ == "__main__":
    main()