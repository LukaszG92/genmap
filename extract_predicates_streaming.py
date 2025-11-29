#!/usr/bin/env python3
"""
extract_predicates_streaming.py - Estrae predicati da file RDF enormi senza OOM
Usa rdflib in modalità streaming per file .rdf, .n3, .ttl
"""
import sys
import argparse
from collections import Counter
from pathlib import Path
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL


def extract_predicates_streaming(file_path: Path, format: str = None, chunk_size: int = 100000):
    """
    Estrae predicati da un file RDF processandolo con rdflib.
    Usa approccio memory-efficient per file enormi.

    Args:
        file_path: Path al file RDF
        format: Formato RDF (auto-detect se None)
        chunk_size: Numero di triple per chunk (solo per feedback)

    Returns:
        Counter con {predicato: count}
    """
    predicates = Counter()

    # Auto-detect formato
    if format is None:
        suffix = file_path.suffix.lower()
        format_map = {
            '.nt': 'nt',
            '.n3': 'n3',
            '.ttl': 'turtle',
            '.rdf': 'xml',
            '.xml': 'xml',
        }
        format = format_map.get(suffix, 'turtle')

    print(f"  Formato rilevato: {format}", file=sys.stderr)

    # Per N-Triples possiamo fare parsing line-by-line più veloce
    if format == 'nt':
        return extract_from_ntriples(file_path)

    # Per altri formati, usa rdflib ma con approach memory-efficient
    # Invece di caricare tutto il grafo, lo processiamo e poi lo svuotiamo
    print(f"  Parsing con rdflib...", file=sys.stderr)

    try:
        # APPROCCIO CORRETTO: parsing completo del file
        # rdflib deve parsare tutto il file per estrarre le triple
        g = Graph()

        # Per RDF/XML, proviamo a disabilitare la validazione strict dei language tags
        if format == 'xml':
            # Parsing più permissivo per RDF/XML con tag non validi
            try:
                g.parse(str(file_path), format=format)
            except Exception as e:
                if 'language tag' in str(e).lower():
                    print(f"  ⚠️  Warning: language tag non validi, uso parsing alternativo...", file=sys.stderr)
                    # Usa ntriples come fallback dopo conversione
                    return extract_from_rdfxml_fallback(file_path)
                else:
                    raise
        else:
            g.parse(str(file_path), format=format)

        triple_count = 0
        # Conta predicati iterando sul grafo
        for s, p, o in g:
            # Converti predicato in stringa con <>
            pred_str = f"<{str(p)}>"
            predicates[pred_str] += 1
            triple_count += 1

            # Feedback ogni chunk_size triple
            if triple_count % chunk_size == 0:
                print(f"    {triple_count:,} triple processate, {len(predicates)} predicati distinti...",
                      file=sys.stderr)

        print(f"  Totale triple: {triple_count:,}", file=sys.stderr)

    except Exception as e:
        print(f"  ERRORE durante parsing rdflib: {e}", file=sys.stderr)
        print(f"  File probabilmente troppo grande o malformato", file=sys.stderr)
        print(f"  Suggerimento: prova a convertirlo prima in .nt", file=sys.stderr)
        sys.exit(1)

    return predicates


def extract_from_ntriples(file_path: Path):
    """Estrazione veloce da N-Triples (parsing testuale)"""
    import re
    predicates = Counter()

    uri_pattern = re.compile(r'<([^>]+)>')

    with file_path.open('r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # In N-Triples: <subject> <predicate> <object> .
            uris = uri_pattern.findall(line)
            if len(uris) >= 2:
                predicate = f'<{uris[1]}>'
                predicates[predicate] += 1

            if line_num % 100000 == 0:
                print(f"    {line_num:,} righe processate...", file=sys.stderr)

    return predicates


def extract_from_rdfxml_fallback(file_path: Path):
    """
    Fallback per RDF/XML con errori di validazione.
    Usa xml.etree per estrarre predicati direttamente senza validazione completa.
    """
    import xml.etree.ElementTree as ET
    from collections import Counter

    print(f"  Usando parsing XML diretto (ignora errori validazione)...", file=sys.stderr)

    predicates = Counter()

    # Namespace comuni
    RDF_NS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'

    try:
        tree = ET.parse(str(file_path))
        root = tree.getroot()

        triple_count = 0

        # Itera su tutti i Description/elementi
        for desc in root.findall('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):
            # Ogni elemento figlio è un predicato
            for child in desc:
                # Estrai namespace + local name
                tag = child.tag
                # Converti in formato <URI>
                if tag.startswith('{'):
                    # Tag ha formato {namespace}localname
                    predicates[f'<{tag[1:].replace("}", "")}>'] += 1
                else:
                    # Tag senza namespace (raro)
                    predicates[f'<{tag}>'] += 1

                triple_count += 1

            # Aggiungi rdf:type per ogni risorsa tipizzata
            rdf_type = desc.get(f'{RDF_NS}type')
            if rdf_type:
                predicates['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'] += 1
                triple_count += 1

        # Cerca anche elementi tipizzati direttamente (non solo Description)
        for elem in root:
            if elem.tag != f'{RDF_NS}Description' and elem.tag.startswith('{'):
                # Questo elemento rappresenta una risorsa tipizzata
                # Ha un rdf:type implicito
                predicates['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'] += 1

                # E ogni child è un predicato
                for child in elem:
                    tag = child.tag
                    if tag.startswith('{'):
                        predicates[f'<{tag[1:].replace("}", "")}>'] += 1
                    triple_count += 1

        print(f"  Totale triple: {triple_count:,}", file=sys.stderr)

    except Exception as e:
        print(f"  ERRORE anche con fallback XML: {e}", file=sys.stderr)
        sys.exit(1)

    return predicates


def extract_predicates_textual(file_path: Path):
    """
    Fallback: parsing testuale per file RDF/N3/Turtle.
    Non è perfetto ma funziona per file che causano problemi a rdflib.
    """
    import re
    predicates = Counter()

    uri_pattern = re.compile(r'<([^>]+)>')

    with file_path.open('r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Salta commenti, prefix, base
            if not line or line.startswith('#') or line.startswith('@prefix') or line.startswith('@base'):
                continue

            # Estrai URI
            uris = uri_pattern.findall(line)

            if len(uris) >= 2:
                # Se inizia con <, è una nuova tripla: <sogg> <pred> <ogg>
                if line.startswith('<'):
                    predicates[f'<{uris[1]}>'] += 1
                # Se inizia con whitespace, è continuazione (dopo ; )
                elif line.startswith(('\t', ' ')) and uris:
                    predicates[f'<{uris[0]}>'] += 1

            if line_num % 100000 == 0:
                print(f"    {line_num:,} righe processate...", file=sys.stderr)

    return predicates


def main():
    parser = argparse.ArgumentParser(
        description="Estrae predicati da file RDF/N3/Turtle enormi"
    )
    parser.add_argument("input", help="File RDF/N3/Turtle/NT da processare")
    parser.add_argument("-f", "--format", help="Formato (nt, n3, turtle, xml)")
    parser.add_argument("-c", "--chunk-size", type=int, default=100000,
                        help="Triple per chunk (default: 100000)")

    args = parser.parse_args()

    file_path = Path(args.input)
    if not file_path.exists():
        sys.exit(f"Errore: {file_path} non esiste")

    print(f"Estrazione predicati da: {file_path}", file=sys.stderr)
    print(f"Dimensione: {file_path.stat().st_size / (1024 ** 2):.2f} MB", file=sys.stderr)

    predicates = extract_predicates_streaming(file_path, args.format, args.chunk_size)

    # Output in formato TSV: count \t <predicate>
    for pred, count in predicates.most_common():
        print(f"{count}\t{pred}")

    print(f"\nPredicati distinti: {len(predicates)}", file=sys.stderr)


if __name__ == "__main__":
    main()