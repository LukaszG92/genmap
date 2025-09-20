# src/genmap/index/build_index.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json, os, re, time
from itertools import islice

from ..config import load_endpoints
from .schema import PredicateInfo, EndpointIndex, FederationIndex, compute_version

# ---------- util ----------

def local_name(iri: str) -> str:
    m = re.search(r'[#/](?!.*[#/])([^#/]+)$', iri)
    return m.group(1) if m else iri

def _try_import_sparql():
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON as SW_JSON, POST
        return SPARQLWrapper, SW_JSON, POST
    except Exception:
        return None, None, None

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# ---------- HARVEST: DISTINCT ?p paginato ----------

def list_predicates_distinct(
    endpoint_url: str,
    page_size: int = 5000,
    timeout_s: int = 30,
    start_after: Optional[str] = None,
) -> List[str]:
    """
    Elenca TUTTI i predicati (?p) in ?s ?p ?o con keyset pagination su STR(?p).
    Itera fino a esaurimento risultati, senza OFFSET (niente SR353).
    - `start_after`: supporta resume (passa l'ultima chiave vista).
    """
    SPARQLWrapper, SW_JSON, POST = _try_import_sparql()
    if SPARQLWrapper is None:
        return []

    s = SPARQLWrapper(endpoint_url)
    s.setReturnFormat(SW_JSON)
    s.setTimeout(timeout_s * 1000)
    s.setMethod(POST)

    preds: List[str] = []
    seen: set[str] = set()  # opzionale: evita duplicati se l'endpoint cambia sotto i piedi
    last_ps: Optional[str] = start_after
    page = 0

    while True:
        page += 1

        if last_ps is None:
            q = f"""
            SELECT ?p ?ps WHERE {{
              ?s ?p ?o .
              BIND(STR(?p) AS ?ps)
            }}
            GROUP BY ?p ?ps
            ORDER BY ?ps
            LIMIT {page_size}
            """
        else:
            from json import dumps as _jdumps
            key = _jdumps(last_ps)
            q = f"""
            SELECT ?p ?ps WHERE {{
              ?s ?p ?o .
              BIND(STR(?p) AS ?ps)
              FILTER(?ps > {key})
            }}
            GROUP BY ?p ?ps
            ORDER BY ?ps
            LIMIT {page_size}
            """

        s.setQuery(q)
        try:
            res = s.query().convert()
        except Exception as e:
            print(f"[warn] seek-page {page} last_ps={last_ps!r}: {e}")
            break

        rows = res.get("results", {}).get("bindings", [])
        if not rows:
            break

        # accumula risultati, con dedup opzionale
        new_count = 0
        for b in rows:
            p = b.get("p", {}).get("value")
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                preds.append(p)
                new_count += 1

        prev_last_ps = last_ps
        last_ps = rows[-1].get("ps", {}).get("value")

        print(f"[seek] page {page}: +{new_count} (raw={len(rows)}) total={len(preds)} last_ps={last_ps!r}")

        # Guardie di terminazione
        if len(rows) < page_size:
            # Esaurimento naturale
            break
        if last_ps == prev_last_ps:
            # Non stiamo avanzando: evitiamo loop infinito
            print("[seek] no progress on keyset, stopping to avoid infinite loop.")
            break

    return preds


# ---------- ENRICH: COUNT + labels/domain/range/equivalents a batch ----------

PREFIXES = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""

def enrich_batch(endpoint_url: str, iris: List[str], timeout_s: int = 60) -> Dict[str, Dict[str, Any]]:
    """
    Per un batch di IRIs, ritorna:
      { iri: { "count": int, "labels": [...], "domain":[...], "range":[...], "equiv":[...] }, ... }
    """
    SPARQLWrapper, SW_JSON, POST = _try_import_sparql()
    if SPARQLWrapper is None or not iris:
        return {}

    values = " ".join(f"<{i}>" for i in iris)
    q = f"""
    {PREFIXES}
    SELECT ?p (COUNT(*) AS ?c)
           (GROUP_CONCAT(DISTINCT ?lbl;separator="||") AS ?labels)
           (GROUP_CONCAT(DISTINCT ?dom;separator="||") AS ?domains)
           (GROUP_CONCAT(DISTINCT ?rng;separator="||") AS ?ranges)
           (GROUP_CONCAT(DISTINCT ?eq;separator="||")  AS ?equivs)
    WHERE {{
      VALUES ?p {{ {values} }}
      ?s ?p ?o .
      OPTIONAL {{ ?p rdfs:label ?lbl FILTER(lang(?lbl)='' || langMatches(lang(?lbl),'en')) }}
      OPTIONAL {{ ?p rdfs:domain ?dom }}
      OPTIONAL {{ ?p rdfs:range  ?rng }}
      OPTIONAL {{ ?p owl:equivalentProperty ?eq }}
    }}
    GROUP BY ?p
    """
    s = SPARQLWrapper(endpoint_url)
    s.setReturnFormat(SW_JSON)
    s.setTimeout(timeout_s * 1000)
    s.setMethod(POST)
    s.setQuery(q)

    out: Dict[str, Dict[str, Any]] = {}
    try:
        res = s.query().convert()
    except Exception as e:
        print(f"[warn] enrich batch failed: {e} (size={len(iris)})")
        return out

    for b in res.get("results", {}).get("bindings", []):
        piri = b["p"]["value"]
        c = int(b["c"]["value"])
        # split fields safely
        def split_field(key):
            v = b.get(key, {}).get("value", "")
            return [x for x in v.split("||") if x] if v else []
        out[piri] = {
            "count": c,
            "labels": split_field("labels"),
            "domains": split_field("domains"),
            "ranges": split_field("ranges"),
            "equivs": split_field("equivs"),
        }
    return out

# ---------- BUILD COMPLETO PER ENDPOINT ----------

def build_endpoint_full(endpoint_id: str, endpoint_url: str,
                        out_json: Path,
                        page_size: int = 5000,
                        batch_size: int = 150,
                        timeout_s: int = 60,
                        resume: bool = True) -> EndpointIndex:
    """
    Costruisce l'indice COMPLETO per un singolo endpoint:
      1) DISTINCT ?p paginato
      2) Enrich in batch con COUNT+metadata
      3) Salvataggio incrementale e resume
    """
    # resume: se esiste un out_json parziale, carica ciò che c'è
    partial_preds: Dict[str, PredicateInfo] = {}
    existing_payload: Optional[Dict[str, Any]] = None
    if resume and out_json.exists():
        try:
            existing_payload = json.loads(out_json.read_text())
            for e in existing_payload.get("endpoints", []):
                if (e.get("id") == endpoint_id) and (e.get("url") == endpoint_url):
                    for p in e.get("predicates", []):
                        pi = PredicateInfo(iri=p["iri"],
                                           local_name=p.get("local_name") or local_name(p["iri"]),
                                           usage_count=p.get("usage_count"),
                                           labels=p.get("labels", []),
                                           domain=p.get("domain", []),
                                           range=p.get("range", []),
                                           equivalents=p.get("equivalents", []))
                        partial_preds[pi.iri] = pi
                    print(f"[resume] recovered {len(partial_preds)} predicates from {out_json}")
        except Exception:
            pass

    # 1) DISTINCT ?p (lista completa)
    print(f"[distinct] scanning all predicates from {endpoint_url} ...")
    all_iris = list_predicates_distinct(endpoint_url, page_size=page_size, timeout_s=timeout_s)
    print(f"[distinct] total predicates: {len(all_iris)}")

    # 2) Enrich a batch
    todo = [iri for iri in all_iris if iri not in partial_preds]
    print(f"[enrich] to process: {len(todo)} (batch_size={batch_size})")

    processed = 0
    t0 = time.time()
    for batch in _chunks(todo, batch_size):
        meta = enrich_batch(endpoint_url, batch, timeout_s=timeout_s)
        for iri in batch:
            pi = partial_preds.get(iri) or PredicateInfo(iri=iri, local_name=local_name(iri))
            m = meta.get(iri, {})
            pi.usage_count = m.get("count", pi.usage_count)
            # append + dedup
            if m.get("labels"):
                pi.labels = sorted(set((pi.labels or []) + m["labels"]))
            if m.get("domains"):
                pi.domain = sorted(set((pi.domain or []) + m["domains"]))
            if m.get("ranges"):
                pi.range = sorted(set((pi.range or []) + m["ranges"]))
            if m.get("equivs"):
                pi.equivalents = sorted(set((pi.equivalents or []) + m["equivs"]))
            partial_preds[iri] = pi

        processed += len(batch)
        if processed % (batch_size*4) == 0 or processed == len(todo):
            # 3) salvataggio incrementale
            ep = EndpointIndex(id=endpoint_id, url=endpoint_url,
                               predicates=list(partial_preds.values()),
                               triples_sampled=None)
            payload = {"endpoints": [ep.model_dump()]}
            ver = compute_version(payload)
            fed = FederationIndex(version=ver, endpoints=[ep])
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(fed.model_dump(), ensure_ascii=False, indent=2))
            dt = time.time() - t0
            print(f"[save] {processed}/{len(todo)} (+{len(batch)}) wrote {out_json} (elapsed {int(dt)}s)")

    # payload finale
    ep = EndpointIndex(id=endpoint_id, url=endpoint_url,
                       predicates=list(partial_preds.values()),
                       triples_sampled=None)
    payload = {"endpoints": [ep.model_dump()]}
    ver = compute_version(payload)
    return FederationIndex(version=ver, endpoints=[ep])

# ---------- ENTRYPOINT MULTI-ENDPOINT (retro-compat) ----------

def build_index(endpoints_file: Path, out_json: Path, limit:int=2000, timeout_s:int=10) -> FederationIndex:
    """
    Manteniamo la firma, ma se l'endpoint è uno solo (DBpedia), usiamo il build completo.
    Se sono più endpoint, eseguiamo il vecchio approccio per ciascuno (con enrich a batch).
    """
    eps = load_endpoints(endpoints_file)
    if len(eps) == 1:
        ep = eps[0]
        fed = build_endpoint_full(ep.get("id") or ep["url"], ep["url"], out_json,
                                  page_size=5000, batch_size=150, timeout_s=max(timeout_s, 30), resume=True)
        out_json.write_text(json.dumps(fed.model_dump(), ensure_ascii=False, indent=2))
        print(f"[done] version={fed.version} wrote={out_json}")
        return fed

    # multi-endpoint: lista distinti + enrich per ciascuno (più veloce di COUNT globale)
    out_eps=[]
    for ep in eps:
        ep_id = ep.get("id") or ep["url"]; ep_url = ep["url"]
        print(f"[multi] endpoint {ep_id} {ep_url}")
        iris = list_predicates_distinct(ep_url, page_size=5000, timeout_s=max(timeout_s,30))
        preds = []
        for batch in _chunks(iris[:limit] if limit else iris, 150):
            meta = enrich_batch(ep_url, batch, timeout_s=max(timeout_s,30))
            for iri in batch:
                m = meta.get(iri, {})
                preds.append(PredicateInfo(
                    iri=iri, local_name=local_name(iri),
                    usage_count=m.get("count"),
                    labels=m.get("labels", []),
                    domain=m.get("domains", []),
                    range=m.get("ranges", []),
                    equivalents=m.get("equivs", [])
                ))
        out_eps.append(EndpointIndex(id=ep_id, url=ep_url, predicates=preds, triples_sampled=None))

    payload = {"endpoints":[e.model_dump() for e in out_eps]}
    ver = compute_version(payload)
    fed = FederationIndex(version=ver, endpoints=out_eps)
    out_json.write_text(json.dumps(fed.model_dump(), ensure_ascii=False, indent=2))
    print(f"[done] version={ver} wrote={out_json}")
    return fed

def main():
    import argparse
    p = argparse.ArgumentParser(description="Build local index (FULL for single endpoint).")
    p.add_argument("-e","--endpoints", default="examples/endpoints.yml", type=Path)
    p.add_argument("-o","--out", default=Path(".cache/index.json"), type=Path)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--page-size", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--max-pages", type=int, default=None, help="debug: stop after N pages of DISTINCT")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    # single endpoint full-build path reads page/batch sizes:
    print(f"[build] endpoints={args.endpoints} → out={args.out} timeout={args.timeout} page={args.page_size} batch={args.batch_size}")
    fed = None
    eps = load_endpoints(args.endpoints)
    if len(eps) == 1:
        ep = eps[0]
        fed = build_endpoint_full(ep.get("id") or ep["url"], ep["url"], args.out,
                                  page_size=args.page_size,
                                  batch_size=args.batch_size,
                                  timeout_s=args.timeout,
                                  resume=args.resume)
        print(f"[done] version={fed.version} wrote={args.out}")
    else:
        fed = build_index(args.endpoints, args.out, limit=None, timeout_s=args.timeout)  # multi
    # stampa finale
    if fed:
        print(f"Index version: {fed.version}")
        print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
