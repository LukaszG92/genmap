# src/genmap/rewrite/rewriter.py (versione finale con raggruppamento completo)
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
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


def _normalize_candidates(candidates: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Normalizza i candidati. Ritorna: {gen:X: {endpoint: predicato}}"""
    if not isinstance(candidates, dict):
        return {}

    if "mapping" in candidates and isinstance(candidates["mapping"], dict):
        root = candidates["mapping"].get("selected", {})
    elif "selected" in candidates:
        root = candidates.get("selected", {})
    else:
        root = candidates

    out: Dict[str, Dict[str, str]] = {}

    if not isinstance(root, dict):
        return out

    for gkey, per_ep in root.items():
        if not isinstance(per_ep, dict):
            continue

        # Assicura che la chiave abbia il prefisso gen:
        gkey = gkey if isinstance(gkey, str) and gkey.startswith("gen:") else f"gen:{gkey}"

        acc: Dict[str, str] = {}
        for ep, val in per_ep.items():
            if val is None:
                continue

            # Estrai il predicato (sempre uno solo)
            if isinstance(val, str):
                acc[ep] = val
            elif isinstance(val, dict):
                pred = val.get("predicate") or val.get("uri") or val.get("iri") or val.get("p") or val.get("property")
                if isinstance(pred, str) and pred:
                    acc[ep] = pred
            elif isinstance(val, list) and len(val) > 0:
                # Prende solo il primo (dovrebbe essere sempre uno solo)
                item = val[0]
                if isinstance(item, str):
                    acc[ep] = item
                elif isinstance(item, dict):
                    pred = item.get("predicate") or item.get("uri") or item.get("iri") or item.get("p") or item.get(
                        "property")
                    if isinstance(pred, str) and pred:
                        acc[ep] = pred

        if acc:
            out[gkey] = acc

    return out


def _endpoint_lookup(endpoints: Any) -> Dict[str, str]:
    """Crea lookup da nome endpoint a URL"""

    def pick_url(d: Dict[str, Any]) -> Optional[str]:
        for key in ("service", "sparql", "url", "endpoint", "iri"):
            v = d.get(key)
            if isinstance(v, str) and v:
                return v
        return None

    lut: Dict[str, str] = {}

    if isinstance(endpoints, dict):
        for k, v in endpoints.items():
            if isinstance(v, dict):
                url = pick_url(v)
                if url:
                    lut[k] = url
            elif isinstance(v, str):
                lut[k] = v
    elif isinstance(endpoints, list):
        for item in endpoints:
            if not isinstance(item, dict):
                continue
            k = item.get("id") or item.get("name") or item.get("endpoint") or item.get("alias")
            url = pick_url(item) or (k if isinstance(k, str) and k.startswith("http") else None)
            if isinstance(k, str) and url:
                lut[k] = url

    return lut


_RE_GEN_TRIPLE = re.compile(
    # soggetto: variabile, IRI o QName tipo prefix:LocalName
    r'(?P<subj>(?:\?[A-Za-z_]\w*|<[^>]+>|[A-Za-z_][\w\-]*:[\w\-]+))\s+'
    # predicato generico
    r'(?P<pred>gen:[A-Za-z_][\w\-]*)\s+'
    # oggetto: variabile, IRI, QName o literal stringa
    r'(?P<obj>(?:\?[A-Za-z_]\w*|<[^>]+>|[A-Za-z_][\w\-]*:[\w\-]+|"[^"]*"(?:@[a-zA-Z\-]+|\^\^<[^>]+>)?))'
    r'(?:\s*\.\s*)?',
    re.MULTILINE
)


def _detect_block_context(body: str, position: int) -> str:
    """
    Rileva in quale contesto si trova una posizione nella query.
    Ritorna: 'main', 'optional', etc.
    """
    before = body[:position]

    # Cerca OPTIONAL non ancora chiusi
    optional_matches = list(re.finditer(r'(?i)\bOPTIONAL\s*\{', before))

    # Per ogni OPTIONAL, controlla se è stato chiuso
    unclosed_optional = False
    for match in optional_matches:
        # Conta le graffe dopo questo OPTIONAL
        after_optional = before[match.end():]
        opens = after_optional.count('{')
        closes = after_optional.count('}')
        if opens >= closes:  # OPTIONAL non ancora chiuso
            unclosed_optional = True
            break

    if unclosed_optional:
        return 'optional'
    else:
        return 'main'


def _tidy(text: str) -> str:
    """Pulisce formattazione SPARQL"""
    text = re.sub(r'\s+([}\)])\s*\.\s*', r' \1', text)
    text = re.sub(r'}\s*{', r'} {', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text


def rewrite(query: str, candidates: Dict[str, Any], endpoints: Any) -> str:
    """
    Riscrive la query sostituendo i predicati generici (gen:X) con quelli selezionati
    e assegnandoli ai SERVICE corretti.

    TUTTE le triple dello stesso (endpoint, context) vengono raggruppate in un unico SERVICE,
    anche se non sono consecutive nella query originale.
    """
    ep_urls = _endpoint_lookup(endpoints)
    cand_map = _normalize_candidates(candidates)
    prologue, body = _split_prologue(query)

    # Trova tutte le triple con predicati generici
    matches = list(_RE_GEN_TRIPLE.finditer(body))

    if not matches:
        out = _fix_spaces_in_prefix_iris(prologue).rstrip() + " " + body.lstrip()
        return _tidy(out)

    # Analizza ogni tripla
    triple_info: List[
        Tuple[int, int, str, str, str, str, str]] = []  # (start, end, subj, pred, obj, endpoint_url, context)

    for m in matches:
        subj = m.group("subj")
        gkey = m.group("pred")
        obj = m.group("obj")

        per_ep = cand_map.get(gkey)
        if not per_ep:
            local = gkey.split(":", 1)[1] if ":" in gkey else gkey
            per_ep = cand_map.get(local) or cand_map.get(f"gen:{local}")

        if not per_ep:
            continue

        ep_name, pred_str = next(iter(per_ep.items()))
        svc_url = ep_urls.get(ep_name)
        if not svc_url:
            if isinstance(ep_name, str) and ep_name.startswith("http"):
                svc_url = ep_name
            else:
                continue

        pred = _sanitize_iri(f"<{pred_str.strip('<> ')}>")
        svc_uri = _sanitize_iri(svc_url).strip("<>")
        context = _detect_block_context(body, m.start())

        triple_info.append((m.start(), m.end(), subj, pred, obj, svc_uri, context))

    if not triple_info:
        out = _fix_spaces_in_prefix_iris(prologue).rstrip() + " " + body.lstrip()
        return _tidy(out)

    # Raggruppa per (endpoint, context)
    groups: Dict[Tuple[str, str], List[int]] = {}

    for i, (start, end, subj, pred, obj, endpoint, context) in enumerate(triple_info):
        key = (endpoint, context)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # Per ogni gruppo, sostituisci tutte le triple con un unico SERVICE nella posizione della prima
    replacements: List[Tuple[int, int, str]] = []

    for (endpoint, context), indices in groups.items():
        # Ordina per posizione
        indices.sort(key=lambda i: triple_info[i][0])

        # Raccogli tutte le triple del gruppo
        triples = []
        positions_to_remove = []

        for i, idx in enumerate(indices):
            start, end, subj, pred, obj, _, _ = triple_info[idx]
            triples.append(f"{subj} {pred} {obj} .")

            if i == 0:
                # Prima tripla: qui metteremo il SERVICE completo
                positions_to_remove.append((start, end, 'replace'))
            else:
                # Triple successive: le rimuoviamo
                positions_to_remove.append((start, end, 'remove'))

        # Crea il SERVICE block
        if len(triples) == 1:
            service_block = f"SERVICE <{endpoint}> {{ {triples[0]} }}"
        else:
            triples_str = "\n    ".join(triples)
            service_block = f"SERVICE <{endpoint}> {{\n    {triples_str}\n  }}"

        # Aggiungi le sostituzioni
        for i, (start, end, action) in enumerate(positions_to_remove):
            if action == 'replace':
                replacements.append((start, end, service_block))
            else:
                # Rimuovi la tripla (sostituisci con stringa vuota)
                replacements.append((start, end, ''))

    # Applica le sostituzioni in ordine inverso (per non invalidare gli offset)
    new_body = body
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_body = new_body[:start] + replacement + new_body[end:]

    # Ricostruisci la query
    if prologue:
        out = _fix_spaces_in_prefix_iris(prologue).rstrip() + "\n" + new_body.lstrip()
    else:
        out = new_body.lstrip()

    return _tidy(out)