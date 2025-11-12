#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from typing import Optional
from rdflib import Graph

# 1) http://bio2rdf.org/<ns>:(ID1 ID2 ...) -> <...:ID1> , <...:ID2> , ...
MULTI_ID_PATTERN = re.compile(
    r'(http://bio2rdf\.org/[A-Za-z0-9._-]+:)'
    r'([^\s<>"\'\)\],;]+(?:\s+[^\s<>"\'\)\],;]+)+)'
)

# 1b) Gestisce URI già wrappati con multi-ID corti: <http://...:<ns>: ID1 ID2>
# Match solo ID alfanumerici brevi (max 20 caratteri, max 5 ID totali)
# per evitare di splittare testo descrittivo come "Acting on the CH-OH group"
WRAPPED_MULTI_ID_PATTERN = re.compile(
    r'<(http://bio2rdf\.org/[A-Za-z0-9._-]+:)\s*'
    r'([A-Za-z0-9._-]{1,20}(?:\s+[A-Za-z0-9._-]{1,20}){1,4})>'
)

# 2) wrappa URI bio2rdf nudi in <...> (evita doppio wrap)
WRAP_URI_PATTERN = re.compile(
    r'(?<!<)(http://bio2rdf\.org/[^\s<>"\'\)\],;]+)'
)

# 3) CHIUDI URI non chiusi anche se seguiti da apice/virgolette/punteggiatura/fine riga
# Permette punti all'interno dell'URI (es. ec:3.6.1.10) e % per URL encoding
UNCLOSED_BIO2RDF = re.compile(
    r'(<http://bio2rdf\.org/[^>\s\'"\u2019\u201D]+)(?=$|\s|[\'"\u2019\u201D]|\s+\.)'
)

# 4) togli apici/virgolette appena PRIMA di un nuovo <http://bio2rdf.org/...>
PRE_URI_GUNK = re.compile(
    r'[\'"\u2019\u201D]+\s*(?=<http://bio2rdf\.org/)'
)

# 5) togli apici/virgolette (anche con spazi) SUBITO DOPO la chiusura di un URI bio2rdf
# Usa lookbehind più specifico per evitare di matchare > dentro literal
POST_URI_GUNK = re.compile(
    r'(?<=bio2rdf\.org/[^\s>]\>)\s*[\'"\u2019\u201D]+(?=$|\s|[,;.\)\]])'
)


def fix_bio2rdf_line(line: str) -> str:
    def repl_multi(m):
        pref = m.group(1)
        ids = m.group(2).split()
        return ' , '.join(f'<{pref}{i}>' for i in ids)

    # ordine importante
    # NOTA: WRAPPED_MULTI_ID_PATTERN è disabilitato perché causa falsi positivi
    # splittando testo descrittivo come "Acting on the CH-OH group"
    # line = WRAPPED_MULTI_ID_PATTERN.sub(repl_multi, line)

    # Gestisci URI nudi con multi-ID
    #line = MULTI_ID_PATTERN.sub(repl_multi, line)
    line = WRAP_URI_PATTERN.sub(r'<\1>', line)

    # FIX: rimuovi doppie aperture << prima di http://bio2rdf.org/
    line = re.sub(r'<<(http://bio2rdf\.org/)', r'<\1', line)
    # FIX: rimuovi doppie chiusure >> dopo URI bio2rdf
    line = re.sub(r'(http://bio2rdf\.org/[^>]+)>>', r'\1>', line)

    # FIX: rimuovi caratteri di punteggiatura dalla FINE degli URI prima di chiuderli
    # Es: <http://...Oxidoreductases;> -> <http://...Oxidoreductases>
    line = re.sub(r'(<http://bio2rdf\.org/[^>]+?)[;,]+>', r'\1>', line)

    # FIX: URL-encode gli spazi e altri caratteri non validi negli URI già wrappati
    # <http://bio2rdf.org/ec:Acting on the CH-OH group> -> <http://bio2rdf.org/ec:Acting%20on%20the%20CH-OH%20group>
    # <http://bio2rdf.org/ec:5'-phospho> -> <http://bio2rdf.org/ec:5%27-phospho>
    def encode_special_chars_in_uri(m):
        uri_content = m.group(1)
        uri_content = uri_content.replace(" ", "%20")
        uri_content = uri_content.replace("'", "%27")
        uri_content = uri_content.replace(""", "%22")
        uri_content = uri_content.replace("'", "%27")  # apostrofo tipografico
        uri_content = uri_content.replace(""", "%22")  # virgoletta tipografica
        return f'<{uri_content}>'

    line = re.sub(r'<(http://bio2rdf\.org/[^>]+)>', encode_special_chars_in_uri, line)

    line = PRE_URI_GUNK.sub('', line)  # ...>'   <http...  -> ...> <http...
    line = UNCLOSED_BIO2RDF.sub(r'\1>', line)  # <http...:NAJ' .  -> <http...:NAJ>' .
    line = POST_URI_GUNK.sub('', line)  # <http...:NAJ>' . -> <http...:NAJ> .

    return line


def fix_file_to_ttl(src: pathlib.Path) -> str:
    out_lines = []
    with src.open('r', encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            out_lines.append(fix_bio2rdf_line(ln))
    return ''.join(out_lines)


def convert_file(in_path: pathlib.Path, out_path: pathlib.Path, base: Optional[str]):
    cleaned_ttl = fix_file_to_ttl(in_path)
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


def main():
    ap = argparse.ArgumentParser(
        description="Ripara .n3 Bio2RDF (spazi/apici) e converte in .nt"
    )
    ap.add_argument("input", help="File .n3 o cartella")
    ap.add_argument("-o", "--out", help="File .nt di output o cartella di output")
    ap.add_argument("--base", help="Base IRI (opzionale)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Sovrascrive file esistenti")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo validazione (non scrive)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        sys.exit(f"Errore: {in_path} non esiste.")

    if in_path.is_file():
        out_path = pathlib.Path(args.out) if args.out else in_path.with_suffix(".nt")
        if out_path.exists() and not args.overwrite and not args.dry_run:
            sys.exit(f"Errore: {out_path} esiste (usa --overwrite).")
        if args.dry_run:
            _ = fix_file_to_ttl(in_path)
            print(f"✓ DRY-RUN ok su {in_path}")
        else:
            convert_file(in_path, out_path, args.base)
            print(f"✓ {in_path} → {out_path}")
    else:
        out_dir = pathlib.Path(args.out) if args.out else in_path
        for f in in_path.rglob("*.n3"):
            rel = f.relative_to(in_path)
            out_path = (out_dir / rel).with_suffix(".nt")
            if out_path.exists() and not args.overwrite and not args.dry_run:
                print(f"⊘ Salto (esiste): {out_path}")
                continue
            if args.dry_run:
                _ = fix_file_to_ttl(f)
                print(f"✓ DRY-RUN ok su {f}")
            else:
                convert_file(f, out_path, args.base)
                print(f"✓ {f} → {out_path}")


if __name__ == "__main__":
    main()