# src/genmap/retrieval/candidates_mock.py
from typing import Dict, List

# Mappa minima di proprietà comuni (solo per la PoC mock)
COMMON: Dict[str, List[str]] = {
    "birthDate": [
        "http://dbpedia.org/ontology/birthDate",
        "http://schema.org/birthDate"
    ],
    "birthPlace": [
        "http://dbpedia.org/ontology/birthPlace",
        "http://schema.org/birthPlace"
    ],
    "name": [
        "http://xmlns.com/foaf/0.1/name",
        "http://schema.org/name",
        "http://dbpedia.org/property/name"
    ],
}

def _suggest_for_token(token: str) -> List[str]:
    # Se nota, usa la lista predefinita; altrimenti crea 1-2 candidati "heuristic"
    if token in COMMON:
        return COMMON[token]
    # fallback molto semplice: un candidato "example", più quello schema.org se plausibile
    suggestions = [f"http://example.org/{token}"]
    # se la prima lettera è minuscola, prova Schema.org (camelCase probabile)
    if token and token[0].islower():
        suggestions.append(f"http://schema.org/{token}")
    return suggestions

def plan_mock_candidates(generics: List[str], endpoints: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Ritorna:
      { "gen:birthDate": {
            "endpointA": [ { "predicate": "...", "score": 0.9, "reason": "mock heuristic" }, ... ],
            "endpointB": [ ... ]
        }, ... }
    """
    out: Dict[str, Dict[str, List[Dict]]] = {}
    for g in generics:
        token = g.split(":", 1)[1] if ":" in g else g
        preds = _suggest_for_token(token)
        per_ep: Dict[str, List[Dict]] = {}
        for ep in endpoints:
            ep_id = ep.get("id") or ep.get("url") or "unknown"
            # Assegna score decrescenti: 0.9, 0.8, ...
            candidates = []
            score = 0.9
            for p in preds:
                candidates.append({
                    "predicate": p,
                    "score": round(score, 3),
                    "reason": f"mock: heuristic for token '{token}'",
                    "source": "mock"
                })
                score -= 0.1
            per_ep[ep_id] = candidates
        out[g] = per_ep
    return out
