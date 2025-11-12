# src/genmap/llm/response_utils.py
from __future__ import annotations
from typing import Dict, Any, List


def convert_array_to_nested_dict(llm_response: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Convert the array-based LLM response to the original nested dictionary format.

    Input (array format):
    {
      "mappings": [
        {
          "generic": "gen:birthDate",
          "endpoint": "https://dbpedia.org/sparql",
          "predicate": "dbo:birthDate",
          "reason": "...",
          "confidence": 0.92
        },
        ...
      ]
    }

    Output (nested dict format):
    {
      "gen:birthDate": {
        "https://dbpedia.org/sparql": {
          "predicate": "dbo:birthDate",
          "reason": "...",
          "confidence": 0.92
        }
      }
    }
    """
    nested = {}

    for mapping in llm_response.get("mappings", []):
        generic = mapping["generic"]
        endpoint = mapping["endpoint"]

        # Initialize nested structure if needed
        if generic not in nested:
            nested[generic] = {}

        # Store the mapping details (excluding generic and endpoint keys)
        nested[generic][endpoint] = {
            "predicate": mapping["predicate"],
            "reason": mapping["reason"],
            "confidence": mapping["confidence"]
        }

    return nested