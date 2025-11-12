# src/genmap/rewrite/rewriter.py  (bridge-aware, optimized)
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional, Set
import re

from sympy import pprint

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
                acc[ep] = [{"predicate": val}]
            elif isinstance(val, dict):
                acc[ep] = [val]
            elif isinstance(val, list):
                lst = [(v if isinstance(v, dict) else {"predicate": str(v)}) for v in val]
                if lst: acc[ep] = lst
        if acc:
            out[gkey] = acc
    return out


def _endpoint_lookup(endpoints: Any) -> Dict[str, str]:
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
    for key in ("predicate", "uri", "iri", "p", "property"):
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
        start = m.start();
        end = m.end()
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


def _get_match_services(m, cand_map, ep_urls) -> Set[str]:
    """Get all possible services for a given match."""
    gkey = m.group("pred")
    per_ep = cand_map.get(gkey)
    if not per_ep:
        local = gkey.split(":", 1)[1] if ":" in gkey else gkey
        per_ep = cand_map.get(local) or cand_map.get(f"gen:{local}")
    if not per_ep:
        return set()

    services = set()
    for ep, lst in per_ep.items():
        if not lst: continue
        pred = _pick_predicate_iri(lst[0])
        if not pred: continue
        svc = ep_urls.get(ep) or (ep if isinstance(ep, str) and ep.startswith("http") else None)
        if svc:
            services.add(svc)
    return services


def _get_vars(m) -> Set[str]:
    """Extract variables from a match."""
    vars = set()
    subj, obj = m.group("subj"), m.group("obj")
    if _is_var(subj):
        vars.add(subj)
    if _is_var(obj):
        vars.add(obj)
    return vars


def _fresh_var(base: str, used: set, tag: str) -> str:
    base = base.lstrip("?")
    i = 1
    while True:
        candidate = f"?{base}_{tag}{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def _segment_by_service_chains(matches, cand_map, ep_urls):
    """
    Segment matches into chains where:
    - If previous match requires UNION of multiple services AND shares variables with next match,
      force a new segment (so results from all services in UNION can flow to next segment)
    - Otherwise, consecutive matches that share ALL services can be grouped together
    Returns segments and bridges between them.
    """
    if not matches:
        return [], []

    segments = []
    bridges = []
    used_vars = set()

    # Start with first match - no renames needed yet
    current_segment = {
        "matches": [matches[0]],
        "services": _get_match_services(matches[0], cand_map, ep_urls),
        "vars": _get_vars(matches[0]),
        "renames": {}
    }
    has_union = len(current_segment["services"]) > 1

    for i in range(1, len(matches)):
        m = matches[i]
        m_services = _get_match_services(m, cand_map, ep_urls)
        m_vars = _get_vars(m)

        shared_services = current_segment["services"] & m_services
        shared_vars = current_segment["vars"] & m_vars

        needs_bridge = False

        if has_union and shared_vars:
            needs_bridge = True
        elif not shared_services and shared_vars:
            needs_bridge = True

        if needs_bridge:
            # Rename shared variables in CURRENT segment for LEFT side of bridge
            for var in shared_vars:
                if var not in current_segment["renames"]:
                    new_var = _fresh_var(var, used_vars, "L")
                    current_segment["renames"][var] = new_var

            segments.append(current_segment)

            # Create new segment with INCOMING renames from bridge (RIGHT side)
            new_segment = {
                "matches": [m],
                "services": m_services,
                "vars": m_vars,
                "renames": {}
            }

            # Build explicit left-alias -> right-canonical mapping for bridge
            bridge_mappings = {}
            for var in shared_vars:
                left_alias = current_segment["renames"][var]
                right_canonical = var
                bridge_mappings[var] = {"left": left_alias, "right": right_canonical}

            bridges.append({
                "left_services": current_segment["services"],
                "right_services": m_services,
                "mappings": bridge_mappings
            })

            current_segment = new_segment
            has_union = len(m_services) > 1
        else:
            # Can add to current segment - inherit any renames
            current_segment["matches"].append(m)
            current_segment["services"] |= m_services
            current_segment["vars"] |= m_vars
            has_union = has_union or len(m_services) > 1

    # Add last segment
    segments.append(current_segment)

    return segments, bridges



def _build_bridge(left_services: Set[str], right_services: Set[str], shared_vars) -> str:
    """
    Build a bridge using ONLY owl:sameAs with explicit left->right variable mappings.
    `shared_vars` can be a dict: { name: {"left": "?o_A", "right": "?o"}, ... } or
    a list of such dicts. We generate, per chosen service (prefer common services):
        SERVICE <svc> {
          { left owl:sameAs right } UNION { right owl:sameAs left }
        } … for each pair
    plus a fallback:
        { BIND(left AS right) … } for each pair
    """
    if not shared_vars:
        return ""

    # Normalize to list of (left, right) pairs
    pairs = []
    if isinstance(shared_vars, dict):
        for _name, mp in shared_vars.items():
            if not isinstance(mp, dict):
                continue
            left = mp.get("left"); right = mp.get("right")
            if isinstance(left, str) and isinstance(right, str):
                pairs.append((left, right))
    elif isinstance(shared_vars, (list, tuple)):
        for mp in shared_vars:
            if not isinstance(mp, dict):
                continue
            left = mp.get("left"); right = mp.get("right")
            if isinstance(left, str) and isinstance(right, str):
                pairs.append((left, right))

    if not pairs:
        return ""

    OWL_SAMEAS = "<http://www.w3.org/2002/07/owl#sameAs>"

    # Prefer services common to both sides, else union of both
    common_services = left_services & right_services
    bridge_services = common_services if common_services else (left_services | right_services)

    blocks = []

    # Service blocks with explicit owl:sameAs in both directions
    for svc in sorted(bridge_services):
        svc_uri = _sanitize_iri(svc).strip("<>")
        parts = [f"SERVICE <{svc_uri}> {{ "]
        for left, right in pairs:
            parts.append(f"{{ {left} {OWL_SAMEAS} {right} }} UNION {{ {right} {OWL_SAMEAS} {left} }} ")
        parts.append("}")
        blocks.append("".join(parts))

    # Fallback: identity propagation if no links found
    binds = " ".join(f"BIND({l} AS {r})" for l, r in pairs)
    if binds:
        blocks.append(binds)

    if not blocks:
        return ""
    if len(blocks) == 1:
        return blocks[0]
    return "{ { " + " } UNION { ".join(blocks) + " } }"

def _build_segment_query(segment, cand_map, ep_urls):
    """Build optimized query for a single segment."""
    matches = segment["matches"]

    # Build per-service triple groups
    service_triples = {}

    for svc in sorted(segment["services"]):
        triples = []
        for m in matches:
            match_services = _get_match_services(m, cand_map, ep_urls)
            if svc not in match_services:
                continue

            subj, gkey, obj = m.group("subj"), m.group("pred"), m.group("obj")
            ren = segment.get("renames", {})
            if subj in ren: subj = ren[subj]
            if obj in ren: obj = ren[obj]
            per_ep = cand_map.get(gkey)
            if not per_ep:
                local = gkey.split(":", 1)[1] if ":" in gkey else gkey
                per_ep = cand_map.get(local) or cand_map.get(f"gen:{local}")
            if not per_ep:
                continue

            # Find the predicate for this service
            for ep, lst in per_ep.items():
                if not lst: continue
                ep_svc = ep_urls.get(ep) or (ep if isinstance(ep, str) and ep.startswith("http") else None)
                if ep_svc == svc:
                    pred = _pick_predicate_iri(lst[0])
                    if pred:
                        triples.append(f"{subj} {pred} {obj} .")
                    break

        if triples:
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for t in triples:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            service_triples[svc] = unique

    if not service_triples:
        return None

    # Build UNION of service calls
    blocks = []
    for svc in sorted(service_triples.keys()):
        svc_uri = _sanitize_iri(svc).strip("<>")
        blocks.append("SERVICE <" + svc_uri + "> { " + " ".join(service_triples[svc]) + " }")

    if len(blocks) == 1:
        return blocks[0]

    return "{ { " + " } UNION { ".join(blocks) + " } }"


def _tidy(text: str) -> str:
    text = re.sub(r'\s+([}\)])\s*\.\s*', r' \1', text)
    text = re.sub(r'}\s*{', r'} {', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text


def _pretty_sparql(text: str) -> str:
    """Indent SPARQL for better readability."""
    lines = text.split('\n')
    result = []
    indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Decrease indent for closing braces
        if stripped.startswith('}'):
            indent = max(0, indent - 1)

        result.append('  ' * indent + stripped)

        # Increase indent for opening braces (but not if closed on same line)
        if '{' in stripped and stripped.count('{') > stripped.count('}'):
            indent += 1

    return '\n'.join(result)


def rewrite(query: str, candidates: Dict[str, Any], endpoints: Any) -> str:
    ep_urls = _endpoint_lookup(endpoints)
    cand_map = _normalize_candidates(candidates)
    prologue, body = _split_prologue(query)

    runs = _collect_contiguous_runs(body)
    if not runs:
        out = _fix_spaces_in_prefix_iris(prologue).rstrip() + " " + body.lstrip()
        return _tidy(out)

    out_parts = []
    last = 0

    for start, end, matches in runs:
        out_parts.append(body[last:start])

        # Segment by service chains
        segments, bridges = _segment_by_service_chains(matches, cand_map, ep_urls)

        # Build query for each segment with bridges between them
        for i, seg in enumerate(segments):
            rewritten = _build_segment_query(seg, cand_map, ep_urls)
            if rewritten:
                rstrip = rewritten.strip()
                if "UNION" in rstrip and not rstrip.startswith("{"):
                    rewritten = "{ " + rstrip + " }"
                out_parts.append(rewritten + " ")

            # Add bridge to next segment if needed
            if i < len(bridges):
                bridge = bridges[i]
                bridge_pattern = _build_bridge(
                    bridge["left_services"],
                    bridge["right_services"],
                    bridge["mappings"]
                )
                if bridge_pattern:
                    bstrip = bridge_pattern.strip()
                    if "UNION" in bstrip and not bstrip.startswith("{"):
                        bridge_pattern = "{ " + bstrip + " }"
                    out_parts.append(bridge_pattern + " ")

        last = end

    out_parts.append(body[last:])
    if prologue:
        out = _fix_spaces_in_prefix_iris(prologue).rstrip() + " " + "".join(out_parts).lstrip()
    else:
        out = "".join(out_parts).lstrip()
    out = _tidy(out)

    if out.count("{") != out.count("}"):
        return _fix_spaces_in_prefix_iris(query)

    return out