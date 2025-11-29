#!/usr/bin/env python3
"""
Helper script semplificato per gestire query e ground truth flat.

Usage:
    python test_helper.py list-queries
    python test_helper.py list-ground-truth
    python test_helper.py add-predicate <gen:pred> <IRI>
    python test_helper.py extract-predicates <query_file>
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Set


def extract_gen_predicates(query: str) -> List[str]:
    """Estrae tutti i predicati gen:* dalla query."""
    pattern = re.compile(r'gen:([A-Za-z_][\w\-]*)')
    matches = pattern.findall(query)
    return sorted(set(f"gen:{m}" for m in matches))


def list_queries(queries_dir: Path):
    """Lista tutte le query nella directory."""
    if not queries_dir.exists():
        print(f"✗ Directory non trovata: {queries_dir}")
        return

    queries = []
    for ext in ["*.txt", "*.rq", "*.sparql"]:
        queries.extend(queries_dir.glob(ext))

    if not queries:
        print(f"⚠ Nessuna query trovata in {queries_dir}")
        return

    print(f"\nQuery trovate in {queries_dir}:")
    print("=" * 80)

    for query_file in sorted(queries):
        query_text = query_file.read_text(encoding="utf-8")
        gen_preds = extract_gen_predicates(query_text)

        print(f"\n📄 {query_file.name}")
        print(f"   ID: {query_file.stem}")
        print(f"   Predicati gen:*: {', '.join(gen_preds) if gen_preds else '(nessuno)'}")
        print(f"   Righe: {len(query_text.splitlines())}")


def list_ground_truth(gt_file: Path):
    """Lista tutti i mapping nel ground truth."""
    if not gt_file.exists():
        print(f"✗ Ground truth non trovato: {gt_file}")
        return

    with gt_file.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Rimuovi chiavi di metadati
    mappings = {k: v for k, v in gt_data.items()
                if not k.startswith("_") and not k.startswith("$")}

    if not mappings:
        print("⚠ Ground truth vuoto")
        return

    print(f"\nGround Truth Mappings ({len(mappings)} totali):")
    print("=" * 80)

    for gen_pred in sorted(mappings.keys()):
        iri = mappings[gen_pred]
        print(f"  {gen_pred:30} → {iri}")


def add_predicate(gen_pred: str, iri: str, gt_file: Path):
    """Aggiunge o aggiorna un mapping nel ground truth."""
    # Assicurati che il predicato inizi con gen:
    if not gen_pred.startswith("gen:"):
        gen_pred = f"gen:{gen_pred}"

    # Carica GT esistente
    if gt_file.exists():
        with gt_file.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)
    else:
        gt_data = {
            "_comment": "Ground Truth semplificato - Mapping flat gen:predicate -> IRI corretto"
        }

    # Aggiorna il mapping
    old_value = gt_data.get(gen_pred)
    gt_data[gen_pred] = iri

    # Salva
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    with gt_file.open("w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    if old_value:
        print(f"✓ Aggiornato: {gen_pred}")
        print(f"  Vecchio: {old_value}")
        print(f"  Nuovo:   {iri}")
    else:
        print(f"✓ Aggiunto: {gen_pred} → {iri}")


def extract_predicates_from_file(query_file: Path, gt_file: Path, interactive: bool = True):
    """Estrae predicati da un file query e propone di aggiungerli al GT."""
    if not query_file.exists():
        print(f"✗ File non trovato: {query_file}")
        return

    query_text = query_file.read_text(encoding="utf-8")
    gen_preds = extract_gen_predicates(query_text)

    if not gen_preds:
        print(f"⚠ Nessun predicato gen:* trovato in {query_file.name}")
        return

    print(f"\nPredicati gen:* trovati in {query_file.name}:")
    for pred in gen_preds:
        print(f"  • {pred}")

    # Carica GT esistente
    if gt_file.exists():
        with gt_file.open("r", encoding="utf-8") as f:
            gt_data = json.load(f)
    else:
        gt_data = {}

    # Identifica predicati mancanti
    missing = [p for p in gen_preds if p not in gt_data]

    if not missing:
        print(f"\n✓ Tutti i predicati sono già nel ground truth")
        return

    print(f"\n⚠ {len(missing)} predicati NON sono nel ground truth:")
    for pred in missing:
        print(f"  • {pred}")

    if interactive:
        print("\n" + "=" * 80)
        print("Vuoi aggiungere questi predicati al ground truth?")
        print("Inserisci l'IRI corretto per ogni predicato (o lascia vuoto per saltare)")
        print("=" * 80)

        for pred in missing:
            print(f"\n{pred}")
            iri = input("  IRI corretto: ").strip()
            if iri:
                gt_data[pred] = iri
                print(f"  ✓ Aggiunto")
            else:
                print(f"  ⊘ Saltato")

        # Salva
        if any(p in gt_data for p in missing):
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            with gt_file.open("w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Ground truth aggiornato: {gt_file}")


def validate_ground_truth(gt_file: Path, queries_dir: Path):
    """Valida il ground truth verificando coverage delle query."""
    if not gt_file.exists():
        print(f"✗ Ground truth non trovato: {gt_file}")
        return

    with gt_file.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Rimuovi chiavi di metadati
    mappings = {k: v for k, v in gt_data.items()
                if not k.startswith("_") and not k.startswith("$")}

    print("\n" + "=" * 80)
    print("VALIDAZIONE GROUND TRUTH")
    print("=" * 80)

    print(f"\nMapping definiti nel GT: {len(mappings)}")

    # Carica tutte le query
    all_gen_preds = set()
    query_files = []

    if queries_dir.exists():
        for ext in ["*.txt", "*.rq", "*.sparql"]:
            query_files.extend(queries_dir.glob(ext))

        for query_file in query_files:
            query_text = query_file.read_text(encoding="utf-8")
            gen_preds = extract_gen_predicates(query_text)
            all_gen_preds.update(gen_preds)

    print(f"Query totali: {len(query_files)}")
    print(f"Predicati gen:* unici nelle query: {len(all_gen_preds)}")

    # Coverage
    covered = all_gen_preds & set(mappings.keys())
    missing = all_gen_preds - set(mappings.keys())
    unused = set(mappings.keys()) - all_gen_preds

    print(f"\n✓ Predicati coperti dal GT: {len(covered)}")
    print(f"⚠ Predicati mancanti nel GT: {len(missing)}")
    print(f"ℹ Predicati nel GT ma non usati: {len(unused)}")

    if missing:
        print(f"\n⚠ Predicati da aggiungere al ground truth:")
        for pred in sorted(missing):
            print(f"  • {pred}")

    if unused:
        print(f"\nℹ Predicati nel GT ma non usati nelle query:")
        for pred in sorted(unused):
            print(f"  • {pred}")

    # Calcola coverage
    if all_gen_preds:
        coverage = (len(covered) / len(all_gen_preds)) * 100
        print(f"\n{'=' * 80}")
        print(f"Coverage: {coverage:.1f}%")
        if coverage == 100:
            print("✓ Ground truth completo!")
        elif coverage >= 80:
            print("⚠ Ground truth quasi completo")
        else:
            print("✗ Ground truth incompleto")


def main():
    parser = argparse.ArgumentParser(
        description="Helper per gestire query e ground truth flat"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire")

    # list-queries
    list_q = subparsers.add_parser("list-queries", help="Lista tutte le query")
    list_q.add_argument("--queries-dir", type=Path, default=Path("test_data_simple/queries"))

    # list-ground-truth
    list_gt = subparsers.add_parser("list-ground-truth", help="Lista tutti i mapping")
    list_gt.add_argument("--ground-truth", type=Path, default=Path("test_data_simple/ground_truth.json"))

    # add-predicate
    add_p = subparsers.add_parser("add-predicate", help="Aggiungi/aggiorna mapping")
    add_p.add_argument("predicate", help="Predicato generico (es. gen:enzyme)")
    add_p.add_argument("iri", help="IRI corretto")
    add_p.add_argument("--ground-truth", type=Path, default=Path("test_data_simple/ground_truth.json"))

    # extract-predicates
    extract_p = subparsers.add_parser("extract-predicates", help="Estrae predicati da query")
    extract_p.add_argument("query_file", type=Path, help="File query")
    extract_p.add_argument("--ground-truth", type=Path, default=Path("test_data_simple/ground_truth.json"))
    extract_p.add_argument("--no-interactive", action="store_true", help="Non chiedere input")

    # validate
    validate_p = subparsers.add_parser("validate", help="Valida ground truth")
    validate_p.add_argument("--ground-truth", type=Path, default=Path("test_data_simple/ground_truth.json"))
    validate_p.add_argument("--queries-dir", type=Path, default=Path("test_data_simple/queries"))

    args = parser.parse_args()

    if args.command == "list-queries":
        list_queries(args.queries_dir)
    elif args.command == "list-ground-truth":
        list_ground_truth(args.ground_truth)
    elif args.command == "add-predicate":
        add_predicate(args.predicate, args.iri, args.ground_truth)
    elif args.command == "extract-predicates":
        extract_predicates_from_file(args.query_file, args.ground_truth,
                                     interactive=not args.no_interactive)
    elif args.command == "validate":
        validate_ground_truth(args.ground_truth, args.queries_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()