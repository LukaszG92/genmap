#!/usr/bin/env python3
"""
Test script semplificato per validare i mapping dei predicati gen:*

Ground Truth Format (flat):
{
  "gen:enzyme": "http://bio2rdf.org/ns/kegg#xEnzyme",
  "gen:description": "http://www4.wiwiss.fu-berlin.de/drugbank/resource/drugbank/description"
}

Usage:
    python test_mappings.py [--server URL] [--queries DIR] [--ground-truth FILE]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import requests


class Colors:
    """ANSI colors for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_queries(queries_dir: Path) -> Dict[str, str]:
    """Carica tutte le query dalla directory (supporta .txt, .rq, .sparql)."""
    queries = {}

    if not queries_dir.exists():
        print(f"{Colors.RED}✗ Directory queries non trovata: {queries_dir}{Colors.END}")
        return queries

    # Supporta .txt, .rq, .sparql
    for ext in ["*.txt", "*.rq", "*.sparql"]:
        for query_file in queries_dir.glob(ext):
            query_id = query_file.stem
            queries[query_id] = query_file.read_text(encoding="utf-8")

    return queries


def load_ground_truth(gt_file: Path) -> Dict[str, str]:
    """
    Carica il ground truth flat da file JSON.

    Formato:
    {
      "gen:enzyme": "http://bio2rdf.org/ns/kegg#xEnzyme",
      "gen:description": "http://..."
    }
    """
    if not gt_file.exists():
        print(f"{Colors.YELLOW}⚠ File ground truth non trovato: {gt_file}{Colors.END}")
        return {}

    with gt_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Se il file ha struttura legacy, prova a convertirlo
    if isinstance(data, dict):
        # Controlla se è già flat (tutte le chiavi iniziano con gen:)
        if all(k.startswith("gen:") or k.startswith("_") or k.startswith("$") for k in data.keys()):
            # Rimuovi chiavi di metadati
            return {k: v for k, v in data.items() if not k.startswith("_") and not k.startswith("$")}

    return data


def extract_gen_predicates(query: str) -> List[str]:
    """Estrae tutti i predicati gen:* dalla query."""
    import re
    pattern = re.compile(r'gen:([A-Za-z_][\w\-]*)')
    matches = pattern.findall(query)
    return [f"gen:{m}" for m in sorted(set(matches))]


def call_translate_api(server_url: str, query: str, timeout: int = 300) -> Dict[str, Any]:
    """Esegue la chiamata all'API /translate."""
    endpoint = f"{server_url}/translate"

    payload = {
        "query": query,
        "endpoints_file": "endpoints/endpoints.yml",
        "index_file": ".cache/index.json",
        "use_sparse": True,
        "use_dense": True
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}✗ Errore nella chiamata API: {e}{Colors.END}")
        return {}


def normalize_iri(iri: str) -> str:
    """Normalizza un IRI rimuovendo < > e spazi."""
    if not iri:
        return ""
    return iri.strip().strip("<>").strip()


def compare_mappings(predicted: Dict[str, Any], expected: Dict[str, str]) -> Dict[str, Any]:
    """
    Confronta i mapping predetti con quelli attesi (ground truth flat).

    Args:
        predicted: Struttura da API {gen:pred -> {endpoint -> {predicate, ...}}}
        expected: Ground truth flat {gen:pred -> IRI_corretto}

    Returns:
        Dizionario con metriche e dettagli
    """
    results = {
        "total_predicates": 0,
        "correct": 0,
        "incorrect": 0,
        "missing": 0,
        "no_ground_truth": 0,
        "details": {}
    }

    # Tutti i predicati da considerare
    all_predicates = set(expected.keys()) | set(predicted.keys())
    results["total_predicates"] = len(all_predicates)

    for gen_pred in sorted(all_predicates):
        expected_iri = expected.get(gen_pred)

        # Se non c'è ground truth per questo predicato
        if not expected_iri:
            results["no_ground_truth"] += 1
            results["details"][gen_pred] = {
                "status": "no_ground_truth",
                "expected": None,
                "predicted": None,
                "message": "Nessun ground truth definito"
            }
            continue

        expected_iri_norm = normalize_iri(expected_iri)

        # Se il predicato non è stato predetto
        if gen_pred not in predicted:
            results["missing"] += 1
            results["details"][gen_pred] = {
                "status": "missing",
                "expected": expected_iri_norm,
                "predicted": None,
                "message": "Predicato non mappato dall'LLM"
            }
            continue

        # Estrai tutti i predicati predetti da tutti gli endpoint
        predicted_iris = []
        endpoints_info = []

        for endpoint, mapping_data in predicted[gen_pred].items():
            if isinstance(mapping_data, dict):
                pred_iri = mapping_data.get("predicate")
            else:
                pred_iri = mapping_data

            if pred_iri:
                pred_iri_norm = normalize_iri(str(pred_iri))
                predicted_iris.append(pred_iri_norm)
                endpoints_info.append({
                    "endpoint": endpoint,
                    "predicate": pred_iri_norm,
                    "match": pred_iri_norm == expected_iri_norm
                })

        # Verifica se almeno uno dei predicati predetti corrisponde
        if expected_iri_norm in predicted_iris:
            results["correct"] += 1
            results["details"][gen_pred] = {
                "status": "correct",
                "expected": expected_iri_norm,
                "predicted": predicted_iris,
                "endpoints": endpoints_info,
                "message": "Mapping corretto trovato"
            }
        else:
            results["incorrect"] += 1
            results["details"][gen_pred] = {
                "status": "incorrect",
                "expected": expected_iri_norm,
                "predicted": predicted_iris,
                "endpoints": endpoints_info,
                "message": "Nessun endpoint ha mappato al predicato corretto"
            }

    return results


def print_report(query_id: str, gen_preds: List[str], results: Dict[str, Any]):
    """Stampa un report dettagliato dei risultati."""
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}Query: {query_id}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.END}")

    print(f"\n{Colors.BOLD}Predicati gen:* trovati:{Colors.END} {', '.join(gen_preds)}")

    # Metriche aggregate
    total = results["total_predicates"]
    correct = results["correct"]
    incorrect = results["incorrect"]
    missing = results["missing"]
    no_gt = results["no_ground_truth"]

    print(f"\n{Colors.BOLD}Metriche:{Colors.END}")
    print(f"  Predicati totali:      {total}")
    print(f"  {Colors.GREEN}✓ Corretti:            {correct}{Colors.END}")
    print(f"  {Colors.RED}✗ Errati:              {incorrect}{Colors.END}")
    #print(f"  {Colors.YELLOW}⚠ Mancanti:            {missing}{Colors.END}")
    print(f"  {Colors.BLUE}? Senza GT:            {no_gt}{Colors.END}")

    # Calcola accuracy (escludendo no_gt)
    evaluated = correct + incorrect + missing
    if evaluated > 0:
        accuracy = (correct / evaluated) * 100
        print(f"\n  {Colors.BOLD}Accuracy: {accuracy:.1f}%{Colors.END}")

    # Dettagli per predicato
    print(f"\n{Colors.BOLD}Dettagli per predicato:{Colors.END}")
    for gen_pred in sorted(results["details"].keys()):
        detail = results["details"][gen_pred]
        status = detail["status"]

        status_icons = {
            "correct": f"{Colors.GREEN}✓{Colors.END}",
            "incorrect": f"{Colors.RED}✗{Colors.END}",
            "missing": f"{Colors.YELLOW}⚠{Colors.END}",
            "no_ground_truth": f"{Colors.BLUE}?{Colors.END}"
        }
        icon = status_icons.get(status, "?")

        print(f"\n  {icon} {Colors.BOLD}{gen_pred}{Colors.END}")
        print(f"      Expected: {detail['expected']}")

        if detail["predicted"]:
            if status == "correct":
                print(f"      {Colors.GREEN}✓ Match trovato{Colors.END}")
            else:
                print(f"      Got: {', '.join(detail['predicted'])}")

            # Mostra dettagli per endpoint
            if detail.get("endpoints"):
                for ep_info in detail["endpoints"]:
                    match_icon = "✓" if ep_info["match"] else "✗"
                    print(f"        {match_icon} {ep_info['endpoint']}: {ep_info['predicate']}")
        else:
            print(f"      {Colors.YELLOW}(nessun mapping predetto){Colors.END}")

        print(f"      → {detail['message']}")


def generate_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Genera un sommario aggregato di tutti i test."""
    summary = {
        "total_queries": len(all_results),
        "total_predicates": 0,
        "total_correct": 0,
        "total_incorrect": 0,
        "total_missing": 0,
        "total_no_gt": 0,
        "queries_perfect": 0,
        "queries_with_errors": 0
    }

    for result in all_results:
        metrics = result["metrics"]
        summary["total_predicates"] += metrics["total_predicates"]
        summary["total_correct"] += metrics["correct"]
        summary["total_incorrect"] += metrics["incorrect"]
        summary["total_missing"] += metrics["missing"]
        summary["total_no_gt"] += metrics["no_ground_truth"]

        if metrics["incorrect"] == 0 and metrics["missing"] == 0:
            summary["queries_perfect"] += 1
        else:
            summary["queries_with_errors"] += 1

    return summary


def print_summary(summary: Dict[str, Any]):
    """Stampa il sommario finale."""
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}SOMMARIO FINALE{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.END}")

    print(f"\nQuery testate:         {summary['total_queries']}")
    print(f"Query perfette:        {Colors.GREEN}{summary['queries_perfect']}{Colors.END}")
    print(f"Query con errori:      {Colors.RED}{summary['queries_with_errors']}{Colors.END}")

    print(f"\nPredicati totali:      {summary['total_predicates']}")
    print(f"{Colors.GREEN}✓ Corretti:            {summary['total_correct']}{Colors.END}")
    print(f"{Colors.RED}✗ Errati:              {summary['total_incorrect']}{Colors.END}")
    #print(f"{Colors.YELLOW}⚠ Mancanti:            {summary['total_missing']}{Colors.END}")
    print(f"{Colors.BLUE}? Senza GT:            {summary['total_no_gt']}{Colors.END}")

    evaluated = summary['total_correct'] + summary['total_incorrect']# + summary['total_missing']
    if evaluated > 0:
        accuracy = (summary['total_correct'] / evaluated) * 100
        print(f"\n{Colors.BOLD}Accuracy complessiva: {accuracy:.1f}%{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description="Test del sistema di mapping gen:* predicates (versione semplificata)"
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="URL del server API (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--queries",
        default="./queries",
        help="Directory contenente le query di test (default: test_data/queries)"
    )
    parser.add_argument(
        "--ground-truth",
        default="./ground_truth.json",
        help="File JSON con i mapping di riferimento (default: test_data/ground_truth.json)"
    )
    parser.add_argument(
        "--output",
        help="Salva i risultati in un file JSON"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Output verboso"
    )

    args = parser.parse_args()

    # Carica queries e ground truth
    queries_dir = Path(args.queries)
    gt_file = Path(args.ground_truth)

    queries = load_queries(queries_dir)
    if not queries:
        print(f"{Colors.RED}✗ Nessuna query trovata in {queries_dir}{Colors.END}")
        sys.exit(1)

    print(f"{Colors.BOLD}Trovate {len(queries)} query da testare{Colors.END}")

    ground_truth = load_ground_truth(gt_file)
    if ground_truth:
        print(f"{Colors.BOLD}Ground truth caricato: {len(ground_truth)} mapping definiti{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ Ground truth vuoto o non trovato{Colors.END}")

    # Test di connessione al server
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        response.raise_for_status()
        print(f"{Colors.GREEN}✓ Server raggiungibile: {args.server}{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ Server non raggiungibile: {e}{Colors.END}")
        sys.exit(1)

    # Esegui i test
    all_results = []

    for query_id, query_text in sorted(queries.items()):
        if args.verbose:
            print(f"\n{Colors.BLUE}→ Testando query: {query_id}{Colors.END}")

        # Estrai predicati gen:*
        gen_preds = extract_gen_predicates(query_text)

        if not gen_preds:
            print(f"{Colors.YELLOW}⚠ Nessun predicato gen:* trovato in {query_id}{Colors.END}")
            continue

        # Chiamata API
        api_response = call_translate_api(args.server, query_text)

        if not api_response:
            print(f"{Colors.RED}✗ Errore nella chiamata API per {query_id}{Colors.END}")
            continue

        # Estrai i mapping predetti
        predicted_mappings = api_response.get("mapping", {}).get("selected", {})

        # Confronta con ground truth
        results = compare_mappings(predicted_mappings, ground_truth)

        # Stampa report
        print_report(query_id, gen_preds, results)

        # Salva risultati
        all_results.append({
            "query_id": query_id,
            "gen_predicates": gen_preds,
            "metrics": results,
            "predicted": predicted_mappings
        })

    # Sommario finale
    if all_results:
        summary = generate_summary(all_results)
        print_summary(summary)

        # Salva output JSON se richiesto
        if args.output:
            output_file = Path(args.output)
            output_data = {
                "summary": summary,
                "results": all_results
            }
            output_file.write_text(
                json.dumps(output_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"\n{Colors.GREEN}✓ Risultati salvati in: {output_file}{Colors.END}")


if __name__ == "__main__":
    main()