# src/genmap/llm/client.py
from __future__ import annotations
from typing import Dict, Any, Optional
import httpx, json

class LLMError(RuntimeError):
    pass

def one_shot_map(base_url: str, model: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    """
    Chiama una sola volta /chat/completions (OpenAI-compat) e ritorna il JSON parsato.
    Se il server non supporta response_format, prova comunque a parse-are content.
    """
    messages = [
        {"role": "system", "content": payload["model_instructions"]},
        {"role": "user",   "content": json.dumps(payload["user_payload"], ensure_ascii=False)}
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }

    try:
        with httpx.Client(base_url=base_url, timeout=timeout_s) as client:
            r = client.post("/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        raise LLMError(str(e)) from e
