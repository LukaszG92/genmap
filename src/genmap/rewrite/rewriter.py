
# src/genmap/rewrite/rewriter.py  (bridge-aware)
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, Set
import re

_RE_PREFIX_LINE = re.compile(r'(?im)^\s*PREFIX\s+([A-Za-z][\w\-]*)\s*:\s*<([^>]+)>\s*$')

def _fix_spaces_in_prefix_iris(text: str) -> str:
    def repl(m):
        prefix, iri = m.group(1), m.group(2)
        return f"PREFIX {prefix}: <{iri.replace(' ', '')}>"
    return _RE_PREFIX_LINE.sub(repl, text)

def _split_prologue(q: str) -> Tuple[str, str]:
    m = re.search(r'(?is)\b(SELECT|CONSTRUCT|ASK|DESCRIBE)\b', q)
    if not m:
        return "", q
    return q[:m.start()], q[m.start():]

def _sanitize_iri(iri: str) -> str:
    iri = iri.strip()
    if iri.startswith("<") and iri.endswith(">"):
        inner = iri[1:-1].replace(" ", "")
        return f"<{inner}>"
    return iri.replace(" ", "")

def _normalize_candidates(candidates: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    if not isinstance(candidates, dict):
        return {}
    if "mapping" in candidates and isinstance(candidates["mapping"], dict):
        root = candidates["mapping"].get("selected", {})
    elif "selected" in candidates:
        root = candidates.get("selected", {})
    else:
        root = candidates
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if not isinstance(root, dict):
        return out
    for gkey, per_ep in root.items():
        if not isinstance(per_ep, dict):
            continue
        gkey = gkey if isinstance(gkey, str) and gkey.startswith("gen:") else f"gen:{gkey}"
        acc: Dict[str, List[Dict[str, Any]]] = {}
        for ep, val in per_ep.items():
            if val is None: continue
            if isinstance(val, str):
                acc[ep] = [ {"predicate": val} ]
            elif isinstance(val, dict):
                acc[ep] = [ val ]
            elif isinstance(val, list):
                lst = [ (v if isinstance(v, dict) else {"predicate": str(v)}) for v in val ]
                if lst: acc[ep] = lst
        if acc:
            out[gkey] = acc
    return out

def _endpoint_lookup(endpoints: Any) -> Dict[str, str]:
    def pick_url(d: Dict[str, Any]) -> Optional[str]:
        for key in ("service","sparql","url","endpoint","iri"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
        return None
    lut: Dict[str, str] = {}
    if isinstance(endpoints, dict):
        for k, v in endpoints.items():
            if isinstance(v, dict):
                url = pick_url(v)
                if url: lut[k] = url
            elif isinstance(v, str):
                lut[k] = v
    elif isinstance(endpoints, list):
        for item in endpoints:
            if not isinstance(item, dict): continue
            k = item.get("id") or item.get("name") or item.get("endpoint") or item.get("alias")
            url = pick_url(item) or (k if isinstance(k, str) and k.startswith("http") else None)
            if isinstance(k, str) and url:
                lut[k] = url
    return lut

def _pick_predicate_iri(c: Dict[str, Any]) -> Optional[str]:
    for key in ("predicate","uri","iri","p","property"):
        v = c.get(key)
        if isinstance(v, str) and v:
            return _sanitize_iri(f"<{v.strip('<> ')}>")
    return None

_RE_GEN_TRIPLE = re.compile(
    r'(?P<subj>(?:\?[A-Za-z_]\w*|<[^>]+>))\s+'
    r'(?P<pred>gen:[A-Za-z_][\w\-]*)\s+'
    r'(?P<obj>(?:\?[A-Za-z_]\w*|<[^>]+>|"[^"]*"(?:@[a-zA-Z\-]+|\^\^<[^>]+>)?))'
    r'(?:\s*\.\s*)?',
    re.MULTILINE
)

def _collect_contiguous_runs(body: str):
    runs = []
    it = list(_RE_GEN_TRIPLE.finditer(body))
    i = 0
    while i < len(it):
        m = it[i]
        start = m.start(); end = m.end()
        group = [m]
        j = i + 1
        while j < len(it):
            nextm = it[j]
            sep = body[end:nextm.start()]
            if sep.strip() == "":
                group.append(nextm)
                end = nextm.end()
                j += 1
            else:
                break
        runs.append((start, end, group))
        i = j
    return runs

def _is_var(tok: str) -> bool:
    return tok.startswith("?")

def _fresh_var(base: str, used: set, tag: str) -> str:
    base = base.lstrip("?")
    i = 1
    while True:
        candidate = f"?{base}_{tag}{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1

def _segment_run_with_bridges(matches):
    segments = []
    bridges = []
    used = set()

    segments.append({"matches": [matches[0]], "renames": {}, "idx": 0})

    for k in range(1, len(matches)):
        prev = matches[k-1]
        cur  = matches[k]
        prev_vars = {v for v in (prev.group("subj"), prev.group("obj")) if _is_var(v)}
        cur_vars  = {v for v in (cur.group("subj"),  cur.group("obj"))  if _is_var(v)}
        shared = list(prev_vars.intersection(cur_vars))

        if not shared:
            segments[-1]["matches"].append(cur)
            continue

        shared_var = shared[0]
        left_seg = segments[-1]
        right_seg_idx = len(segments)
        right_seg = {"matches": [cur], "renames": {}, "idx": right_seg_idx}
        segments.append(right_seg)

        left_local = left_seg["renames"].get(shared_var)
        if not left_local:
            left_local = _fresh_var(shared_var, used, "A")
            left_seg["renames"][shared_var] = left_local
        right_local = right_seg["renames"].get(shared_var)
        if not right_local:
            right_local = _fresh_var(shared_var, used, "B")
            right_seg["renames"][shared_var] = right_local

        bridges.append({
            "left_idx": left_seg["idx"],
            "right_idx": right_seg["idx"],
            "orig_var": shared_var,
            "left_var": left_local,
            "right_var": right_local,
        })

    return segments, bridges

def _apply_renames(token: str, renames: Dict[str,str]) -> str:
    return renames.get(token, token) if token.startswith("?") else token

def _build_union_group_for_segment(seg, cand_map, ep_urls):
    ep_triples = {}
    used_services = set()
    ren = seg["renames"]

    for m in seg["matches"]:
        subj, gkey, obj = m.group("subj"), m.group("pred"), m.group("obj")
        subj = _apply_renames(subj, ren)
        obj  = _apply_renames(obj, ren)

        per_ep = cand_map.get(gkey)
        if not per_ep:
            local = gkey.split(":",1)[1] if ":" in gkey else gkey
            per_ep = cand_map.get(local) or cand_map.get(f"gen:{local}")
        if not per_ep:
            continue

        for ep, lst in per_ep.items():
            if not lst: continue
            pred = _pick_predicate_iri(lst[0])
            if not pred: continue
            svc = ep_urls.get(ep) or (ep if isinstance(ep, str) and ep.startswith("http") else None)
            if not svc: continue
            used_services.add(svc)
            ep_triples.setdefault(svc, []).append(f"{subj} {pred} {obj} .")

    if not ep_triples:
        return None, set()

    blocks = []
    for svc in sorted(ep_triples.keys()):
        triples = ep_triples[svc]
        seen = set(); uniq = []
        for t in triples:
            if t not in seen:
                seen.add(t); uniq.append(t)
        svc_uri = _sanitize_iri(svc).strip("<>")
        blocks.append("{ SERVICE <" + svc_uri + "> { " + " ".join(uniq) + " } }")

    blocks = list(dict.fromkeys(blocks))
    return "{ " + " UNION ".join(blocks) + " }", used_services

def _build_bridge_block(left_var: str, right_var: str, service_urls: list[str]) -> str:
    """
    Ponte generico tra 'left_var' e 'right_var' usando owl:sameAs / skos:exactMatch
    (senza SERVICE SILENT), includendo sia il verso diretto che quello inverso
    nello *stesso* SERVICE per ogni endpoint.
    Restituisce un group pattern del tipo:
      {
        { SERVICE <EP1> { VALUES ?p { <owl#sameAs> <skos#exactMatch> }
                          { ?L ?p ?R . } UNION { ?R ?p ?L . } } }
        UNION
        { SERVICE <EP2> { ... } }
        UNION
        { BIND(?L AS ?R) }
      }
    """
    OWL_SAMEAS = "<http://www.w3.org/2002/07/owl#sameAs>"
    SKOS_EXACT = "<http://www.w3.org/2004/02/skos/core#exactMatch>"

    blocks = []
    for svc in sorted(service_urls):
        svc_uri = _sanitize_iri(svc).strip("<>")
        blocks.append(
            "{ SERVICE <" + svc_uri + "> {\n"
            "    VALUES ?p { " + OWL_SAMEAS + " " + SKOS_EXACT + " }\n"
            "    { " + left_var + " ?p " + right_var + " . }\n"
            "    UNION\n"
            "    { " + right_var + " ?p " + left_var + " . }\n"
            "} }"
        )

    # Fallback: se nessun link esiste, permetti comunque l'uguaglianza
    blocks.append("{ BIND(" + left_var + " AS " + right_var + ") }")
    return "{\n  " + "\n  UNION\n  ".join(blocks) + "\n}"


def _tidy(text: str) -> str:
    text = re.sub(r'\s+([}\)])\s*\.\s*', r' \1', text)
    text = re.sub(r'}\s*{', r'} {', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text

def _pretty_sparql(text: str) -> str:
    """Indenta leggermente gruppi/UNION/ SERVICE per rendere più leggibile l'output."""
    # linea per UNION, SERVICE e graffe
    text = re.sub(r'\{\s*SERVICE', r'{\n  SERVICE', text)
    text = re.sub(r'\}\s*UNION\s*\{', r'}\n  UNION\n{', text)
    text = re.sub(r'\}\s*\}', r'}\n}', text)
    # compatta spazi multipli
    text = re.sub(r'[ \t]+(\n)', r'\1', text)
    return text

def rewrite(query: str, candidates: Dict[str, Any], endpoints: Any) -> str:
    ep_urls = _endpoint_lookup(endpoints)
    cand_map = _normalize_candidates(candidates)
    prologue, body = _split_prologue(query)

    runs = _collect_contiguous_runs(body)
    if not runs:
        out = _fix_spaces_in_prefix_iris(prologue) + body
        return _tidy(out)

    out_parts = []
    last = 0
    for start, end, matches in runs:
        out_parts.append(body[last:start])

        segments, bridges = _segment_run_with_bridges(matches)

        seg_groups = []
        seg_services = []
        for seg in segments:
            grp, used = _build_union_group_for_segment(seg, cand_map, ep_urls)
            if grp:
                seg_groups.append(grp)
                seg_services.append(used)

        for i, grp in enumerate(seg_groups):
            out_parts.append(" " + grp + " ")
            if i < len(seg_groups) - 1 and i < len(bridges):
                b = bridges[i]
                services = set()
                if i < len(seg_services): services |= seg_services[i]
                if i+1 < len(seg_services): services |= seg_services[i+1]
                bridge = _build_bridge_block(b["left_var"], b["right_var"], sorted(services))
                out_parts.append(" " + bridge + " ")

        last = end

    out_parts.append(body[last:])
    out = _fix_spaces_in_prefix_iris(prologue) + "".join(out_parts)
    out = _tidy(out)

    if out.count("{") != out.count("}"):
        return _fix_spaces_in_prefix_iris(query)

    print(_pretty_sparql(out))
    return out
