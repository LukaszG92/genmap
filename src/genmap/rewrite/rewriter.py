# src/genmap/rewrite/rewriter.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import re

import re

def _tidy_punctuation(q: str) -> str:
    # Rimuovi il punto subito dopo la chiusura di un SERVICE o di una UNION (...).
    q = re.sub(r"\}\s*\.\s*", "}", q)   #  ... } .  ->  ... }
    q = re.sub(r"\)\s*\.\s*", ")", q)   #  ... ) .  ->  ... )
    return q


# Head di una chain: s p o ;|.
RE_STMT_HEAD = re.compile(
    r'\s*(?P<s>[^\s\{\};.]+)\s+(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])',
    re.MULTILINE
)
# Continuazioni col soggetto sottinteso: p o ;|.
RE_STMT_CONT = re.compile(
    r'\s*(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])',
    re.MULTILINE
)

def _endpoint_lookup(endpoints: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ep in endpoints:
        ep_id = ep.get("id") or ep.get("url") or "unknown"
        out[ep_id] = ep.get("url", ep_id)
    return out

def _pick_top1_per_endpoint(candidates_for_gen: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """ candidates[ep_id] = [ {predicate, score, ...}, ... ] → { ep_id: predicate_top1 } """
    choice: Dict[str, str] = {}
    for ep_id, lst in candidates_for_gen.items():
        if not lst:
            continue
        pred = lst[0].get("predicate")
        if pred:
            choice[ep_id] = pred
    return choice

def _union_block_for_triple(s: str, o: str, gen_pred: str,
                            candidates: Dict[str, Any],
                            ep_urls: Dict[str, str]) -> str | None:
    by_ep = candidates.get(gen_pred, {})
    top1 = _pick_top1_per_endpoint(by_ep)
    blocks: List[str] = []
    for ep_id, pred_uri in top1.items():
        ep_url = ep_urls.get(ep_id, ep_id)
        blocks.append(f"SERVICE <{ep_url}> {{ {s} <{pred_uri}> {o} . }}")
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    return "( " + " ) UNION ( ".join(blocks) + " )"

def _consume_chain(query: str, head_match) -> Tuple[List[Tuple[str,str,str]], int]:
    """Dato un match di head, consuma eventuali continuazioni ';' e
       ritorna ([(s,p,o), ...], end_pos) della chain.
    """
    s = head_match.group("s")
    p = head_match.group("p")
    o = head_match.group("o")
    sep = head_match.group("sep")
    triples = [(s, p, o)]
    pos = head_match.end()

    while sep == ';':
        mc = RE_STMT_CONT.match(query, pos)
        if not mc:
            break
        p2 = mc.group("p")
        o2 = mc.group("o")
        sep = mc.group("sep")
        triples.append((s, p2, o2))
        pos = mc.end()

    return triples, pos

def rewrite(query: str, candidates: Dict[str, Any], endpoints: List[Dict[str, Any]]) -> str:
    """Riscrive le chain col ';' espandendo ogni tripla e
       raggruppando triple consecutive che vanno allo stesso endpoint.
       (No pushdown FILTER né grouping multi-BGP per ora.)
    """
    ep_urls = _endpoint_lookup(endpoints)

    def flush_group(group_ep_id: str, group_triples: List[Tuple[str,str,str]], out: List[str]):
        """Emette un singolo SERVICE con più triple per lo stesso endpoint."""
        if not group_ep_id or not group_triples:
            return
        ep_url = ep_urls.get(group_ep_id, group_ep_id)
        inside = " ".join(f"{s} <{pred}> {o} ." for (s, pred, o) in group_triples)
        out.append(f"SERVICE <{ep_url}> {{ {inside} }}")

    out_parts: List[str] = []
    last = 0
    L = len(query)

    while True:
        mh = RE_STMT_HEAD.search(query, last)
        if not mh:
            break

        out_parts.append(query[last:mh.start()])  # testo precedente invariato
        triples, chain_end = _consume_chain(query, mh)

        # grouping locale alla chain
        group_ep_id: str | None = None
        group_triples: List[Tuple[str,str,str]] = []  # (s, predIRI, o)

        def close_group():
            nonlocal group_ep_id, group_triples
            flush_group(group_ep_id, group_triples, out_parts)
            group_ep_id, group_triples = None, []

        for (s, p, o) in triples:
            if p.startswith("gen:"):
                # prova a costruire UNION/top1
                by_ep = candidates.get(p, {})
                top1 = _pick_top1_per_endpoint(by_ep)

                if not top1:
                    # nessun candidato → lascia la tripla com'è, ma prima chiudi il gruppo
                    close_group()
                    out_parts.append(f"{s} {p} {o} .")
                    continue

                if len(top1) == 1:
                    # candidato unico → ottimo per grouping
                    ep_id, pred_iri = next(iter(top1.items()))
                    if group_ep_id is None:
                        # apre un nuovo gruppo
                        group_ep_id = ep_id
                        group_triples = [(s, pred_iri, o)]
                    elif group_ep_id == ep_id:
                        # continua nel gruppo corrente
                        group_triples.append((s, pred_iri, o))
                    else:
                        # altro endpoint: chiudi gruppo corrente e aprine uno nuovo
                        close_group()
                        group_ep_id = ep_id
                        group_triples = [(s, pred_iri, o)]
                else:
                    # piu' endpoint → UNION: chiudi gruppo e emetti blocco UNION singolo per questa tripla
                    close_group()
                    union_block = _union_block_for_triple(s, o, p, candidates, ep_urls)
                    if union_block is not None:
                        out_parts.append(f"{union_block}")
                    else:
                        out_parts.append(f"{s} {p} {o} .")
            else:
                # tripla non-gen → chiudi gruppo e scrivi com'è
                close_group()
                out_parts.append(f"{s} {p} {o} .")

        # fine chain: chiudi eventuale gruppo rimasto aperto
        close_group()
        last = chain_end

    out_parts.append(query[last:L])
    text = "".join(out_parts)
    return _tidy_punctuation(text)
