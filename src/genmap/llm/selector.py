# src/genmap/llm/selector.py
from __future__ import annotations
from typing import Dict, Any, List

def selected_from_llm(generics: List[str],
                      endpoints: List[dict],
                      llm_json: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    allowed_eps = {ep.get("id") or ep.get("url") for ep in endpoints}
    mappings = (llm_json or {}).get("mappings", {})

    for g in generics:
        per_ep = {}
        gmap = mappings.get(g, {}) if isinstance(mappings.get(g, {}), dict) else {}
        for ep_id, obj in gmap.items():
            if ep_id not in allowed_eps:
                continue
            pred = obj.get("predicate")
            conf = float(obj.get("confidence", 0) or 0)
            reason = obj.get("reason", "llm")
            per_ep[ep_id] = ([{"predicate": pred, "score": conf, "reason": reason, "source": "llm"}] if pred else [])
        out[g] = per_ep
    return out
