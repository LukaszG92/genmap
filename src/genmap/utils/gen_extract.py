import re
from typing import Dict, List

RE_GEN_TOKEN = re.compile(r'(?<![\w:])gen:([A-Za-z_][\w\-]*)')
RE_STMT_HEAD = re.compile(r'\s*(?P<s>[^\s\{\}]+)\s+(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])', re.MULTILINE)
RE_STMT_CONT = re.compile(r'\s*(?P<p>[^\s\{\};.]+)\s+(?P<o>[^\s\{\};.]+)\s*(?P<sep>[;.])', re.MULTILINE)
RE_GEN_PRED = re.compile(r'^gen:[A-Za-z_][\w\-]*$')

def extract_gen_predicates(query: str) -> Dict[str, List[str]]:
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

        if s.upper() in {"PREFIX", "BASE"}:
            pos = mh.end()
            continue

        if RE_GEN_PRED.match(p):
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

            if RE_GEN_PRED.match(p2):
                if p2 not in order:
                    order[p2] = len(order)
                triples.append(f"{s} {p2} {o2}")

        # se la head chiude con '.', si passa alla prossima statement chain
        # (pos è già su mh.end() o mc.end())

    predicates = [p for p, _ in sorted(order.items(), key=lambda kv: kv[1])]
    return {"predicates": predicates, "triples": triples}
