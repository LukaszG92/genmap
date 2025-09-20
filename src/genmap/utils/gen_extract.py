# src/genmap/utils/gen_extract.py
import re
from typing import Dict, List

# Trova token gen:* ovunque (utile per catturare anche usi non in posizione di predicato)
RE_GEN_TOKEN = re.compile(r'(?<![\w:])gen:([A-Za-z_][\w\-]*)')

# Head di una "statement chain": s p o ;|.
RE_STMT_HEAD = re.compile(
    r'\s*(?P<s>[^\s\{\}]+)\s+(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])',
    re.MULTILINE
)
# Continuazioni col soggetto sottinteso:    p o ;|.
RE_STMT_CONT = re.compile(
    r'\s*(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])',
    re.MULTILINE
)

def extract_gen_predicates(query: str) -> Dict[str, List[str]]:
    """
    Ritorna:
      - 'predicates': lista unica di gen:* nell'ordine di prima apparizione
      - 'triples':    triple testuali 's gen:pred o' catturate sia come head che come continuazioni
    Nota: parser "best-effort" (regex). Non copre tutti i casi SPARQL (es. literal con spazi).
    """
    # 0) colleziona tutti i token gen:* per ordine di apparizione (anche se non sono predicati)
    order: Dict[str, int] = {}
    for m in RE_GEN_TOKEN.finditer(query):
        tok = f"gen:{m.group(1)}"
        if tok not in order:
            order[tok] = len(order)

    triples: List[str] = []
    pos = 0
    L = len(query)

    while pos < L:
        mh = RE_STMT_HEAD.search(query, pos)
        if not mh:
            break

        s = mh.group("s")
        p = mh.group("p")
        o = mh.group("o")
        sep = mh.group("sep")

        # se head ha un gen:* come predicato, registra la tripla
        if p.startswith("gen:"):
            if p not in order:
                order[p] = len(order)
            triples.append(f"{s} {p} {o}")

        pos = mh.end()

        # se la head termina con ';', cattura le continuazioni col soggetto sottinteso
        while sep == ';':
            mc = RE_STMT_CONT.match(query, pos)
            if not mc:
                break
            p2 = mc.group("p")
            o2 = mc.group("o")
            sep = mc.group("sep")
            pos = mc.end()

            if p2.startswith("gen:"):
                if p2 not in order:
                    order[p2] = len(order)
                triples.append(f"{s} {p2} {o2}")

        # se la head chiude con '.', si passa alla prossima statement chain
        # (pos è già su mh.end() o mc.end())

    predicates = [p for p, _ in sorted(order.items(), key=lambda kv: kv[1])]
    return {"predicates": predicates, "triples": triples}
