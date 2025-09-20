# src/genmap/llm/schema.py
from __future__ import annotations

def genmap_json_schema():
    """
    JSON Schema per Structured Outputs:
    {
      "mappings": {
        "gen:birthDate": {
          "endpointA": {"predicate": "... or null", "reason": "...", "confidence": 0..1},
          "endpointB": {...}
        },
        ...
      }
    }
    """
    return {
        "name": "GenMapMappingsV1",
        "strict": True,  # forza aderenza allo schema
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mappings": {
                    "type": "object",
                    # chiavi dinamiche per ogni gen:* → ognuna è un oggetto di endpoint
                    "additionalProperties": {
                        "type": "object",
                        # chiavi dinamiche per ogni endpointId → ognuna è l’oggetto scelta
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "predicate": {"type": ["string", "null"]},
                                "reason":    {"type": "string"},
                                "confidence":{"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["predicate", "reason", "confidence"]
                        }
                    }
                }
            },
            "required": ["mappings"]
        }
    }
