# src/genmap/api/server.py
import os
import logging
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..utils.gen_extract import extract_gen_predicates
from ..config import load_endpoints
from ..rewrite.rewriter import rewrite
from ..config import Settings
from ..index.search_candidates import search_candidates
from ..llm.openai_client import one_shot_map_openai

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("genmap.api")

# Per logging più dettagliato durante sviluppo, puoi cambiare a DEBUG
logger.setLevel(logging.DEBUG)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="genmap PoC – step6")


class TranslateIn(BaseModel):
    query: str
    endpoints_file: str = "examples/endpoints.yml"
    index_file: str = ".cache/index.json"
    use_sparse: bool = True
    use_dense: bool = False
    popa: float = 0.0
    fusion_mode: Optional[str] = None  # "rrf" | "wsum"; None = usa Settings
    fusion_alpha: Optional[float] = None


class TranslateOut(BaseModel):
    mapping: dict
    rewritten: str


@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok", "step": 6}


@app.post("/translate", response_model=TranslateOut)
def translate(body: TranslateIn):
    """
    Endpoint principale per la traduzione di query SPARQL con predicati generici.
    """
    request_id = f"{int(time.time() * 1000)}"
    start_time = time.time()

    logger.info(f"[{request_id}] ========== NEW TRANSLATION REQUEST ==========")
    logger.info(f"[{request_id}] Query length: {len(body.query)} chars")
    logger.debug(f"[{request_id}] Query: {body.query[:200]}...")  # primi 200 char

    try:
        # =====================================================================
        # 1. CARICAMENTO SETTINGS
        # =====================================================================
        logger.info(f"[{request_id}] Loading settings...")
        settings = Settings()

        use_sparse = getattr(body, "use_sparse", True)
        use_dense = getattr(body, "use_dense", False)
        fusion_mode = getattr(body, "fusion_mode", None) or getattr(settings, "fusion_mode", None)
        fusion_alpha = getattr(body, "fusion_alpha", None) or getattr(settings, "fusion_alpha", None)
        popa = getattr(body, "popa", 0.0) or getattr(settings, "popa", 0.0)

        logger.info(f"[{request_id}] Parameters:")
        logger.info(f"[{request_id}]   - use_sparse: {use_sparse}")
        logger.info(f"[{request_id}]   - use_dense: {use_dense}")
        logger.info(f"[{request_id}]   - fusion_mode: {fusion_mode}")
        logger.info(f"[{request_id}]   - fusion_alpha: {fusion_alpha}")
        logger.info(f"[{request_id}]   - popa: {popa}")
        logger.info(f"[{request_id}]   - endpoints_file: {body.endpoints_file}")
        logger.info(f"[{request_id}]   - index_file: {body.index_file}")

        # =====================================================================
        # 2. ESTRAZIONE PREDICATI GENERICI
        # =====================================================================
        logger.info(f"[{request_id}] Extracting generic predicates...")
        step_start = time.time()

        info = extract_gen_predicates(body.query)
        generics = info["predicates"]

        step_duration = time.time() - step_start
        logger.info(f"[{request_id}] Found {len(generics)} generic predicates in {step_duration:.3f}s")
        logger.info(f"[{request_id}] Generic predicates: {generics}")

        if not generics:
            logger.warning(f"[{request_id}] No generic predicates found in query")
            return TranslateOut(
                mapping={"candidates": {}, "selected": {}, "params": {}},
                rewritten=body.query
            )

        # =====================================================================
        # 3. CARICAMENTO ENDPOINTS
        # =====================================================================
        logger.info(f"[{request_id}] Loading endpoints from {body.endpoints_file}...")
        step_start = time.time()

        endpoints_path = Path('./endpoints/endpoints.yml')
        if not endpoints_path.exists():
            logger.error(f"[{request_id}] Endpoints file not found: {endpoints_path}")
            raise HTTPException(status_code=404, detail=f"Endpoints file not found: {endpoints_path}")

        endpoints = load_endpoints(endpoints_path)
        step_duration = time.time() - step_start

        endpoint_count = len(endpoints) if isinstance(endpoints, (list, dict)) else 0
        logger.info(f"[{request_id}] Loaded {endpoint_count} endpoints in {step_duration:.3f}s")

        # =====================================================================
        # 4. RICERCA CANDIDATI PER OGNI PREDICATO GENERICO
        # =====================================================================
        logger.info(f"[{request_id}] Searching candidates for {len(generics)} predicates...")
        step_start = time.time()

        candidates = {}
        for i, g in enumerate(generics, 1):
            logger.debug(f"[{request_id}] Searching candidates for predicate {i}/{len(generics)}: {g}")
            pred_start = time.time()

            per_endpoint = search_candidates(g, popa=popa)
            candidates[g] = per_endpoint

            pred_duration = time.time() - pred_start
            total_candidates = sum(len(eps) if isinstance(eps, list) else 1 for eps in per_endpoint.values())
            logger.debug(
                f"[{request_id}]   -> Found {len(per_endpoint)} endpoints with {total_candidates} total candidates in {pred_duration:.3f}s")

        step_duration = time.time() - step_start
        logger.info(f"[{request_id}] Candidate search completed in {step_duration:.3f}s")

        # Log statistiche candidati
        total_endpoints = sum(len(per_ep) for per_ep in candidates.values())
        total_cands = sum(
            sum(len(eps) if isinstance(eps, list) else 1 for eps in per_ep.values())
            for per_ep in candidates.values()
        )
        logger.info(f"[{request_id}] Total: {total_endpoints} endpoint matches, {total_cands} candidates")

        # =====================================================================
        # 5. MAPPING CON LLM
        # =====================================================================
        logger.info(f"[{request_id}] Calling LLM for predicate mapping...")
        step_start = time.time()

        selected = one_shot_map_openai("gpt-5-chat-latest", body.query, generics, candidates, timeout_s=300)

        step_duration = time.time() - step_start
        logger.info(f"[{request_id}] LLM mapping completed in {step_duration:.3f}s")
        logger.debug(f"[{request_id}] Selected mappings: {selected}")

        # =====================================================================
        # 6. VALIDAZIONE E FILTRAGGIO
        # =====================================================================
        logger.info(f"[{request_id}] Validating selected mappings...")

        def _valid_gen_key(k: str) -> bool:
            if ":" not in k:
                return False
            pref, local = k.split(":", 1)
            return bool(pref) and bool(local.strip())

        original_count = len(selected)
        selected = {k: v for k, v in selected.items() if _valid_gen_key(k)}
        filtered_count = original_count - len(selected)

        if filtered_count > 0:
            logger.warning(f"[{request_id}] Filtered out {filtered_count} invalid mappings")

        logger.info(f"[{request_id}] Valid mappings: {len(selected)}")
        for gen_pred, mapping in selected.items():
            if isinstance(mapping, dict):
                endpoint_count = len(mapping)
                logger.debug(f"[{request_id}]   {gen_pred} -> {endpoint_count} endpoint(s)")

        # =====================================================================
        # 7. REWRITE DELLA QUERY
        # =====================================================================
        logger.info(f"[{request_id}] Rewriting query...")
        step_start = time.time()

        rewritten = rewrite(body.query, selected, endpoints)

        step_duration = time.time() - step_start
        logger.info(f"[{request_id}] Query rewrite completed in {step_duration:.3f}s")
        logger.info(f"[{request_id}] Rewritten query length: {len(rewritten)} chars")
        logger.debug(f"[{request_id}] Rewritten query: {rewritten[:300]}...")

        # =====================================================================
        # 8. PREPARAZIONE RISPOSTA
        # =====================================================================
        mapping = {
            "selected": selected,
            "params": {
                "use_sparse": use_sparse,
                "use_dense": use_dense,
                "dense_model": getattr(settings, "dense_model", None),
                "fusion_mode": fusion_mode,
                "fusion_alpha": fusion_alpha
            }
        }

        # =====================================================================
        # 9. FINE REQUEST
        # =====================================================================
        total_duration = time.time() - start_time
        logger.info(f"[{request_id}] ========== REQUEST COMPLETED in {total_duration:.3f}s ==========")

        return TranslateOut(mapping=mapping, rewritten=rewritten)

    except Exception as e:
        logger.error(f"[{request_id}] ========== REQUEST FAILED ==========")
        logger.error(f"[{request_id}] Error type: {type(e).__name__}")
        logger.error(f"[{request_id}] Error message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")