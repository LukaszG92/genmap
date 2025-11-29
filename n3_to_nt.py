#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from typing import Optional, TextIO
from rdflib import Graph

# Stessi pattern del file originale
MULTI_ID_PATTERN = re.compile(
    r'(http://bio2rdf\.org/[A-Za-z0-9._-]+:)'
    r'([^\s<>"\'\)\],;]+(?:\s+[^\s<>"\'\)\],;]+)+)'
)

WRAPPED_MULTI_ID_PATTERN = re.compile(
    r'<(http://bio2rdf\.org/[A-Za-z0-9._-]+:)\s*'
    r'([A-Za-z0-9._-]{1,20}(?:\s+[A-Za-z0-9._-]{1,20}){1,4})>'
)

WRAP_URI_PATTERN = re.compile(
    r'(?<!<)(http://bio2rdf\.org/[^\s<>"\'\)\],;]+)'
)

UNCLOSED_BIO2RDF = re.compile(
    r'(<http://bio2rdf\.org/[^>\s\'"\u2019\u201D]+)(?=$|\s|[\'"\u2019\u201D]|\s+\.)'
)

PRE_URI_GUNK = re.compile(
    r'[\'"\u2019\u201D]+\s*(?=<http://bio2rdf\.org/)'
)

POST_URI_GUNK = re.compile(
    r'(?<=bio2rdf\.org/[^\s>]\>)\s*[\'"\u2019\u201D]+(?=$|\s|[,;.\)\]])'
)


def fix_bio2rdf_line(line: str) -> str:
    def repl_multi(m):
        pref = m.group(1)
        ids = m.group(2).split()
        return ' , '.join(f'<{pref}{i}>' for i in ids)

    line = WRAP_URI_PATTERN.sub(r'<\1>', line)
    line = re.sub(r'<<(http://bio2rdf\.org/)', r'<\1', line)
    line = re.sub(r'(http://bio2rdf\.org/[^>]+)>>', r'\1>', line)
    line = re.sub(r'(<http://bio2rdf\.org/[^>]+?)[;,]+>', r'\1>', line)

    def encode_special_chars_in_uri(m):
        uri_content = m.group(1)
        uri_content = uri_content.replace(" ", "%20")
        uri_content = uri_content.replace("'", "%27")
        uri_content = uri_content.replace(""", "%22")
        uri_content = uri_content.replace("'", "%27")
        uri_content = uri_content.replace(""", "%22")
        return f'<{uri_content}>'

    line = re.sub(r'<(http://bio2rdf\.org/[^>]+)>', encode_special_chars_in_uri, line)
    line = PRE_URI_GUNK.sub('', line)
    line = UNCLOSED_BIO2RDF.sub(r'\1>', line)
    line = POST_URI_GUNK.sub('', line)

    return line


def process_large_file_streaming(in_path: pathlib.Path, out_path: pathlib.Path,
                                 chunk_size: int = 10000, base: Optional[str] = None):
    """
    Processa un file enorme dividendolo in chunk più piccoli.
    IMPORTANTE: divide i chunk solo alla fine di un statement completo (al punto finale).
    """
    print(f"Processamento in modalità streaming (chunk di ~{chunk_size} righe)...")

    # Crea il file di output (lo sovrascrive se esiste)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_lines = []
    chunk_num = 0
    total_lines = 0
    total_triples = 0
    prefixes = []  # Memorizza i prefix per includerli in ogni chunk
    in_prefix_section = True

    with in_path.open('r', encoding='utf-8', errors='replace') as fh:
        for line_num, line in enumerate(fh, 1):
            # Applica le correzioni
            fixed_line = fix_bio2rdf_line(line)
            total_lines += 1

            # Cattura le dichiarazioni @prefix per includerle in ogni chunk
            stripped = fixed_line.strip()
            if stripped.startswith('@prefix') or stripped.startswith('@base'):
                if in_prefix_section:
                    prefixes.append(fixed_line)
                    continue  # Non contare i prefix nel chunk
                else:
                    # Prefix dichiarati dopo: aggiungi alla lista ma includi anche nel chunk
                    prefixes.append(fixed_line)
            elif stripped and not stripped.startswith('#'):
                # Prima riga non-prefix: fine della sezione prefix
                in_prefix_section = False

            chunk_lines.append(fixed_line)

            # Controlla se la riga termina uno statement (finisce con punto)
            # NOTA: in Turtle/N3, uno statement termina con '.' eventualmente seguito da commento
            ends_statement = False
            if stripped.endswith('.') or ('. #' in stripped and stripped.rstrip().endswith('.')):
                # Verifica che non sia parte di un URI o literal
                # Un vero statement-ending point è seguito da niente o whitespace/commento
                if stripped[-1] == '.' or stripped.rstrip().endswith('.'):
                    ends_statement = True

            # Quando raggiungiamo chunk_size E siamo alla fine di uno statement, processiamo
            if len(chunk_lines) >= chunk_size and ends_statement:
                chunk_num += 1
                # Prepara chunk con i prefix
                full_chunk = prefixes + chunk_lines
                triples = process_chunk(full_chunk, out_path, chunk_num, base,
                                        append=(chunk_num > 1))
                total_triples += triples
                print(f"  Chunk {chunk_num}: {total_lines} righe processate, "
                      f"{triples} triple in questo chunk, "
                      f"{total_triples} triple totali")
                chunk_lines = []

    # Processa l'ultimo chunk se ci sono righe rimanenti
    if chunk_lines:
        chunk_num += 1
        full_chunk = prefixes + chunk_lines
        triples = process_chunk(full_chunk, out_path, chunk_num, base,
                                append=(chunk_num > 1))
        total_triples += triples
        print(f"  Chunk {chunk_num} (finale): {total_lines} righe processate, "
              f"{triples} triple in questo chunk, "
              f"{total_triples} triple totali")

    print(f"\n✓ Completato: {total_lines} righe → {total_triples} triple in {out_path}")
    return total_triples


def process_chunk(lines: list, out_path: pathlib.Path, chunk_num: int,
                  base: Optional[str], append: bool = False):
    """
    Processa un chunk di righe: parsing + serializzazione.
    Se append=True, appende al file esistente, altrimenti lo crea.
    """
    chunk_text = ''.join(lines)
    g = Graph()

    try:
        g.parse(data=chunk_text, format="turtle", publicID=base)
    except Exception as e:
        # Salva il chunk problematico per debug
        debug_file = out_path.with_suffix(f".chunk{chunk_num}.debug.ttl")
        pathlib.Path(debug_file).write_text(chunk_text, encoding="utf-8")
        raise RuntimeError(
            f"Parsing fallito sul chunk {chunk_num}.\n"
            f"Ho salvato il chunk in: {debug_file}\n"
            f"Dettagli: {e}"
        )

    # Serializza in N-Triples
    mode = 'ab' if append else 'wb'  # append binary o write binary
    with open(out_path, mode) as f:
        f.write(g.serialize(format="nt", encoding="utf-8"))

    return len(g)


def convert_file_small(in_path: pathlib.Path, out_path: pathlib.Path, base: Optional[str]):
    """Versione originale per file piccoli (tutto in memoria)"""
    out_lines = []
    with in_path.open('r', encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            out_lines.append(fix_bio2rdf_line(ln))

    cleaned_ttl = ''.join(out_lines)
    g = Graph()
    try:
        g.parse(data=cleaned_ttl, format="turtle", publicID=base)
    except Exception as e:
        dbg = out_path.with_suffix(".cleaned.ttl")
        pathlib.Path(dbg).write_text(cleaned_ttl, encoding="utf-8")
        raise RuntimeError(
            f"Parsing fallito su {in_path}.\n"
            f"Ho salvato il Turtle riparato in: {dbg}\n"
            f"Dettagli: {e}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="nt", encoding="utf-8")


def get_file_size_mb(path: pathlib.Path) -> float:
    """Ritorna la dimensione del file in MB"""
    return path.stat().st_size / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser(
        description="Ripara .n3 Bio2RDF (spazi/apici) e converte in .nt (gestisce file enormi)"
    )
    ap.add_argument("input", help="File .n3 o cartella")
    ap.add_argument("-o", "--out", help="File .nt di output o cartella di output")
    ap.add_argument("--base", help="Base IRI (opzionale)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Sovrascrive file esistenti")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo validazione (non scrive)")
    ap.add_argument("--chunk-size", type=int, default=10000,
                    help="Numero di righe per chunk (default: 10000)")
    ap.add_argument("--size-threshold", type=int, default=100,
                    help="Soglia in MB per usare modalità streaming (default: 100)")
    ap.add_argument("--force-streaming", action="store_true",
                    help="Forza modalità streaming anche per file piccoli")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        sys.exit(f"Errore: {in_path} non esiste.")

    if in_path.is_file():
        out_path = pathlib.Path(args.out) if args.out else in_path.with_suffix(".nt")
        if out_path.exists() and not args.overwrite and not args.dry_run:
            sys.exit(f"Errore: {out_path} esiste (usa --overwrite).")

        file_size_mb = get_file_size_mb(in_path)
        print(f"Dimensione file: {file_size_mb:.2f} MB")

        if args.dry_run:
            print("Modalità DRY-RUN: solo validazione sintattica delle prime righe...")
            chunk_lines = []
            with in_path.open('r', encoding='utf-8', errors='replace') as fh:
                for i, line in enumerate(fh):
                    if i >= 1000:  # Valida solo le prime 1000 righe
                        break
                    chunk_lines.append(fix_bio2rdf_line(line))
            print(f"✓ DRY-RUN ok su {in_path} (prime 1000 righe)")
        else:
            # Usa streaming se il file è grande o se forzato
            if args.force_streaming or file_size_mb > args.size_threshold:
                process_large_file_streaming(in_path, out_path, args.chunk_size, args.base)
                # Elimina il file .n3 dopo conversione riuscita
                try:
                    in_path.unlink()
                    print(f"🗑️  Eliminato file originale: {in_path}")
                except Exception as e:
                    print(f"⚠️  Errore eliminazione {in_path}: {e}")
            else:
                print("File piccolo: uso modalità standard (tutto in memoria)")
                convert_file_small(in_path, out_path, args.base)
                print(f"✓ {in_path} → {out_path}")
                # Elimina il file .n3
                try:
                    in_path.unlink()
                    print(f"🗑️  Eliminato: {in_path}")
                except Exception as e:
                    print(f"⚠️  Errore eliminazione {in_path}: {e}")

    else:
        # Gestione cartella
        out_dir = pathlib.Path(args.out) if args.out else in_path
        all_files = list(in_path.rglob("*.n3"))
        total_files = len(all_files)

        if total_files == 0:
            print(f"Nessun file .n3 trovato in {in_path}")
            return

        print(f"Trovati {total_files} file .n3 da processare\n")

        for idx, f in enumerate(all_files, start=1):
            rel = f.relative_to(in_path)
            out_path = (out_dir / rel).with_suffix(".nt")

            file_size_mb = get_file_size_mb(f)

            if out_path.exists() and not args.overwrite and not args.dry_run:
                print(f"[{idx}/{total_files}] ⊘ Salto (esiste): {out_path}")
                try:
                    f.unlink()
                    print(f"                    🗑️  Eliminato: {f}")
                except Exception as e:
                    print(f"                    ⚠️  Errore eliminazione {f}: {e}")
                continue

            if args.dry_run:
                print(f"[{idx}/{total_files}] ✓ DRY-RUN ok su {f} ({file_size_mb:.2f} MB)")
            else:
                print(f"[{idx}/{total_files}] Processamento {f} ({file_size_mb:.2f} MB)...")

                if args.force_streaming or file_size_mb > args.size_threshold:
                    process_large_file_streaming(f, out_path, args.chunk_size, args.base)
                else:
                    convert_file_small(f, out_path, args.base)
                    print(f"                    ✓ {f} → {out_path}")

                # Elimina il file .n3
                try:
                    f.unlink()
                    print(f"                    🗑️  Eliminato: {f}")
                except Exception as e:
                    print(f"                    ⚠️  Errore eliminazione {f}: {e}")


if __name__ == "__main__":
    main()