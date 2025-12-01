#!/usr/bin/env python3
"""
Test script per validare il sistema di traduzione predicati gen:*

Usage:
    python test_system.py --server http://localhost:8000 --queries . --ground-truth gen_mappings.json
"""

import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        print(f"{desc}...", file=sys.stderr)
        return iterable


def load_queries(queries_dir: Path) -> Dict[str, str]:
    """Carica tutte le query dalla directory."""
    queries = {}
    if not queries_dir.exists():
        return queries

    for query_file in queries_dir.glob("*.tex"):
        query_id = query_file.stem
        queries[query_id] = query_file.read_text(encoding="utf-8")

    return queries


def load_ground_truth(gt_file: Path) -> Dict[str, str]:
    """Carica il ground truth da file JSON."""
    if not gt_file.exists():
        return {}

    with gt_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {k: v for k, v in data.items()
            if not k.startswith("_") and not k.startswith("$")}


def extract_gen_predicates(query: str) -> List[str]:
    """Estrae tutti i predicati gen:* dalla query."""
    pattern = re.compile(r'gen:([A-Za-z_][\w\-]*)')
    matches = pattern.findall(query)
    return sorted(set(f"gen:{m}" for m in matches))


def call_translate_api(server_url: str, query: str, popa: float = None, timeout: int = 300) -> Tuple[Dict, float]:
    """
    Esegue chiamata all'API /translate e misura il tempo.

    Args:
        server_url: URL del server
        query: Query SPARQL
        popa: Parametro popularity adjustment (opzionale)
        timeout: Timeout in secondi

    Returns:
        (response_dict, execution_time_seconds)
    """
    endpoint = f"{server_url}/translate"

    payload = {
        "query": query,
        "endpoints_file": "endpoints/endpoints.yml",
        "index_file": ".cache/index.json",
        "use_sparse": True,
        "use_dense": True
    }

    # Aggiungi popa se specificato
    if popa is not None:
        payload["popa"] = popa

    start_time = time.time()

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        response.raise_for_status()
        execution_time = time.time() - start_time
        return response.json(), execution_time
    except requests.exceptions.RequestException:
        execution_time = time.time() - start_time
        return {}, execution_time


def normalize_iri(iri: str) -> str:
    """Normalizza un IRI."""
    if not iri:
        return ""
    return iri.strip().strip("<>").strip()


def evaluate_mappings(predicted: Dict[str, Any], expected: Dict[str, str], query_predicates: List[str]) -> Tuple[
    Dict[str, int], List[Dict[str, str]]]:
    """
    Valuta i mapping predetti vs ground truth per i predicati della query.

    Args:
        predicted: Mapping predetti dall'API
        expected: Ground truth completo
        query_predicates: Lista dei predicati gen:* presenti nella query

    Returns:
        (metrics, errors)
        metrics: {
            "total": int,
            "correct": int,
            "incorrect": int,
            "replaced": int
        }
        errors: [
            {
                "gen_predicate": str,
                "expected": str,
                "predicted": str
            }
        ]
    """
    results = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "replaced": 0
    }

    errors = []

    # Valuta SOLO i predicati della query che hanno ground truth
    all_predicates = [p for p in query_predicates if p in expected]
    results["total"] = len(all_predicates)

    for gen_pred in all_predicates:
        expected_iri = normalize_iri(expected[gen_pred])

        # Se il predicato non è stato sostituito
        if gen_pred not in predicted:
            results["incorrect"] += 1
            errors.append({
                "gen_predicate": gen_pred,
                "expected": expected_iri,
                "predicted": "(non sostituito)"
            })
            continue

        results["replaced"] += 1

        # Estrai il predicato predetto
        # Struttura: selected[gen:pred] = { "endpoint_name": { "predicate": "...", "reason": "...", "confidence": X } }
        mapping_data = predicted[gen_pred]

        if isinstance(mapping_data, dict):
            # Prendi il primo (e unico) endpoint
            endpoint_data = next(iter(mapping_data.values()))
            if isinstance(endpoint_data, dict):
                pred_iri = endpoint_data.get("predicate")
            else:
                pred_iri = endpoint_data
        else:
            pred_iri = mapping_data

        if not pred_iri:
            results["incorrect"] += 1
            errors.append({
                "gen_predicate": gen_pred,
                "expected": expected_iri,
                "predicted": "(vuoto)"
            })
            continue

        predicted_iri = normalize_iri(str(pred_iri))

        # Verifica se corrisponde
        if predicted_iri == expected_iri:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
            errors.append({
                "gen_predicate": gen_pred,
                "expected": expected_iri,
                "predicted": predicted_iri
            })

    return results, errors


def main():
    parser = argparse.ArgumentParser(description="Test sistema traduzione gen:*")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="URL server API")
    parser.add_argument("--queries", default="./queries", help="Directory query")
    parser.add_argument("--ground-truth", default="gen_mappings.json", help="File ground truth")
    parser.add_argument("--output", help="Salva risultati JSON")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in secondi tra le chiamate (default: 1.0)")

    args = parser.parse_args()

    # Carica dati
    queries_dir = Path(args.queries)
    gt_file = Path(args.ground_truth)

    queries = load_queries(queries_dir)
    if not queries:
        print(f"✗ Nessuna query trovata in {queries_dir}")
        sys.exit(1)

    ground_truth = load_ground_truth(gt_file)
    if not ground_truth:
        print(f"✗ Ground truth non trovato: {gt_file}")
        sys.exit(1)

    # Test connessione
    try:
        requests.get(f"{args.server}/health", timeout=5)
    except Exception as e:
        print(f"✗ Server non raggiungibile: {e}")
        sys.exit(1)

    # Esegui test con popa=0
    all_results = []
    all_errors = []
    total_time = 0.0
    time_per_predicate_list = []

    for query_id, query_text in tqdm(sorted(queries.items()), desc="Testing queries"):
        gen_preds = extract_gen_predicates(query_text)

        if not gen_preds:
            continue

        # Chiamata API con popa=0
        api_response, exec_time = call_translate_api(args.server, query_text, popa=0)
        total_time += exec_time

        if not api_response:
            continue

        # Estrai mapping predetti
        predicted_mappings = api_response.get("mapping", {}).get("selected", {})

        # Valuta
        metrics, errors = evaluate_mappings(predicted_mappings, ground_truth, gen_preds)

        # Calcola tempo per predicato per questa query
        if metrics["replaced"] > 0:
            time_per_pred = exec_time / metrics["replaced"]
            time_per_predicate_list.append(time_per_pred)

        # Aggiungi query_id agli errori
        for error in errors:
            error["query_id"] = query_id
        all_errors.extend(errors)

        all_results.append({
            "query_id": query_id,
            "gen_predicates": gen_preds,
            "execution_time": exec_time,
            "metrics": metrics
        })

        # Delay tra chiamate (non conteggiato nel tempo di esecuzione)
        if args.delay > 0:
            time.sleep(args.delay)

    # Calcola metriche aggregate
    if not all_results:
        print("✗ Nessun risultato da mostrare")
        sys.exit(1)

    total_queries = len(all_results)
    avg_time = total_time / total_queries if total_queries > 0 else 0
    avg_time_per_predicate = sum(time_per_predicate_list) / len(
        time_per_predicate_list) if time_per_predicate_list else 0

    total_correct = sum(r["metrics"]["correct"] for r in all_results)
    total_incorrect = sum(r["metrics"]["incorrect"] for r in all_results)
    total_predicates = sum(r["metrics"]["total"] for r in all_results)
    total_replaced = sum(r["metrics"]["replaced"] for r in all_results)

    evaluated = total_correct + total_incorrect
    precision = (total_correct / evaluated * 100) if evaluated > 0 else 0

    # Conta errori per tipo
    error_counter = Counter()
    for error in all_errors:
        error_key = f"{error['gen_predicate']}: expected={error['expected']}, got={error['predicted']}"
        error_counter[error_key] += 1

    # Report finale
    print("\n" + "=" * 70)
    print("REPORT FINALE")
    print("=" * 70)
    print(f"\nQuery testate:              {total_queries}")
    print(f"Predicati totali:           {total_predicates}")
    print(f"Predicati sostituiti:       {total_replaced}")
    print(f"Mapping corretti:           {total_correct}")
    print(f"Mapping errati:             {total_incorrect}")
    print(f"\nPrecisione:                 {precision:.2f}%")
    print(f"Tempo medio per query:      {avg_time:.2f}s")
    print(f"Tempo medio per predicato:  {avg_time_per_predicate:.3f}s")
    print(f"Tempo totale:               {total_time:.2f}s")

    # Report errori
    if error_counter:
        print("\n" + "=" * 70)
        print(f"ERRORI DI MAPPING ({len(all_errors)} totali)")
        print("=" * 70)

        for error_msg, count in error_counter.most_common():
            print(f"\n[{count}x] {error_msg}")

    print("=" * 70)

    # Salva output
    if args.output:
        output_data = {
            "summary": {
                "total_queries": total_queries,
                "total_predicates": total_predicates,
                "total_replaced": total_replaced,
                "total_correct": total_correct,
                "total_incorrect": total_incorrect,
                "precision": precision,
                "avg_execution_time": avg_time,
                "avg_time_per_predicate": avg_time_per_predicate,
                "total_time": total_time
            },
            "results": all_results,
            "errors": all_errors,
            "error_summary": [{"error": k, "count": v} for k, v in error_counter.most_common()]
        }

        Path(args.output).write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n✓ Risultati salvati in: {args.output}")


if __name__ == "__main__":
    main()