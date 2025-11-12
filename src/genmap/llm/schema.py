# src/genmap/llm/schema.py
from __future__ import annotations


def genmap_json_schema():
    """
    JSON Schema per Structured Outputs.

    OpenAI's strict mode has limitations with additionalProperties.
    We use a wrapper array structure to work around this:
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
    """
    return {
        "name": "GenMapMappingsV1",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "generic": {"type": "string"},
                            "endpoint": {"type": "string"},
                            "predicate": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"}
                                ]
                            },
                            "reason": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["generic", "endpoint", "predicate", "reason", "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["mappings"],
            "additionalProperties": False
        }
    }