# src/genmap/llm/openai_client.py
from __future__ import annotations
from typing import Dict, Any, List
import json
from openai import OpenAI
from openai import APIError, APIStatusError, RateLimitError, AuthenticationError, BadRequestError

from .prompt import build_messages
from .schema import genmap_json_schema
from .response_utils import convert_array_to_nested_dict


class OpenAIError(RuntimeError):
    pass


def _select_highest_confidence(nested_mappings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per ogni predicato generico, seleziona solo il mapping con la confidence più alta.

    Input: {
        "gen:predicate1": {
            "endpoint1": {"predicate": "...", "confidence": 0.8, ...},
            "endpoint2": {"predicate": "...", "confidence": 0.95, ...}
        }
    }

    Output: {
        "gen:predicate1": {
            "endpoint2": {"predicate": "...", "confidence": 0.95, ...}
        }
    }
    """
    result = {}

    for gen_pred, endpoints_dict in nested_mappings.items():
        if not isinstance(endpoints_dict, dict):
            continue

        # Trova l'endpoint con la confidence più alta
        best_endpoint = None
        best_confidence = -1.0
        best_mapping = None

        for endpoint_name, mapping in endpoints_dict.items():
            if not isinstance(mapping, dict):
                continue

            # Estrai la confidence (default 0.0 se non presente)
            confidence = mapping.get("confidence", 0.0)

            # Converti a float se è stringa o altro
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0

            # Aggiorna il migliore se questa confidence è più alta
            if confidence > best_confidence:
                best_confidence = confidence
                best_endpoint = endpoint_name
                best_mapping = mapping

        # Aggiungi solo il mapping migliore
        if best_endpoint is not None and best_mapping is not None:
            result[gen_pred] = {best_endpoint: best_mapping}

    return result


def one_shot_map_openai(model: str,
                        query: str,
                        generics: List[str],
                        candidates: Dict[str, Dict[str, List[Dict[str, Any]]]],
                        timeout_s: int = 60) -> Dict[str, Any]:
    """
    Chiama l'LLM per mappare predicati generici a predicati reali.
    Restituisce solo il mapping con confidence più alta per ogni predicato generico.
    """
    client = OpenAI(timeout=timeout_s)
    messages = build_messages(query, generics, candidates)

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_schema", "json_schema": genmap_json_schema()},
            messages=messages,
        )
        content = resp.choices[0].message.content
        llm_response = json.loads(content)

        # Convert array format to nested dictionary format
        nested_mappings = convert_array_to_nested_dict(llm_response)

        # Seleziona solo il mapping con confidence più alta per ogni predicato
        filtered_mappings = _select_highest_confidence(nested_mappings)

        return filtered_mappings

    except (BadRequestError, AuthenticationError, RateLimitError, APIStatusError, APIError) as e:
        # Errore da API OpenAI (400/401/429/5xx). Convertilo in OpenAIError con dettagli utili.
        raise OpenAIError(f"OpenAI API error: {repr(e)}") from e
    except Exception as e:
        # Qualsiasi altro errore (network, parse, ecc.)
        raise OpenAIError(f"LLM call failed: {repr(e)}") from e