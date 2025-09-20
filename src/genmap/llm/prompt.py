# src/genmap/llm/prompt.py
from __future__ import annotations
from typing import Dict, Any, List
import json

def build_messages(query: str,
                   generics: List[str],
                   candidates: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> list[dict]:
    system = (
        "You are an expert SPARQL/ontology assistant. "
        "For each generic predicate and for each endpoint, pick at most ONE predicate from the provided candidates, "
        "or NONE if unsure. Do NOT invent IRIs. Return ONLY JSON per the schema."
    )
    user_payload = {
        "query": query,
        "generics": generics,
        "candidates": candidates,
        "rules": [
            "Use ONLY the provided candidates for each (generic, endpoint).",
            "At most one predicate per endpoint.",
            "If unsure, set predicate = null and explain briefly in 'reason'."
        ]
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]