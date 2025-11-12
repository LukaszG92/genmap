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


def one_shot_map_openai(model: str,
                        query: str,
                        generics: List[str],
                        candidates: Dict[str, Dict[str, List[Dict[str, Any]]]],
                        timeout_s: int = 60) -> Dict[str, Any]:
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

        return nested_mappings
    except (BadRequestError, AuthenticationError, RateLimitError, APIStatusError, APIError) as e:
        # Errore da API OpenAI (400/401/429/5xx). Convertilo in OpenAIError con dettagli utili.
        raise OpenAIError(f"OpenAI API error: {repr(e)}") from e
    except Exception as e:
        # Qualsiasi altro errore (network, parse, ecc.)
        raise OpenAIError(f"LLM call failed: {repr(e)}") from e