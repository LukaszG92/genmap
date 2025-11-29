#!/usr/bin/env bash
# build_predicates_rdf.sh — lavora direttamente su .rdf/.n3 senza conversione
# Usa extract_predicates_streaming.py per parsing corretto con rdflib
set -euo pipefail

IN_DIR="${1:-./dumps/SWDFood}"
OUT_DIR="${2:-./predicates/swdFood}"
ENDPOINT_ID="${3:-swdFood}"

# Cerca lo script Python di estrazione
EXTRACTOR=""
if [ -f "./extract_predicates_streaming.py" ]; then
  EXTRACTOR="./extract_predicates_streaming.py"
elif [ -f "$(dirname "$0")/extract_predicates_streaming.py" ]; then
  EXTRACTOR="$(dirname "$0")/extract_predicates_streaming.py"
else
  echo "ERRORE: extract_predicates_streaming.py non trovato!" >&2
  echo "Assicurati che extract_predicates_streaming.py sia nella stessa directory di questo script" >&2
  exit 1
fi

echo "Usando extractor: $EXTRACTOR"

PER_FILE_DIR="$OUT_DIR/files"
mkdir -p "$PER_FILE_DIR"

# prepara lista file con glob, supporta .rdf, .n3, .ttl, .nt
shopt -s nullglob
FILES=( "$IN_DIR"/*.nt "$IN_DIR"/*.nt.gz "$IN_DIR"/*.rdf "$IN_DIR"/*.n3 "$IN_DIR"/*.ttl )
TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
  printf "Nessun file RDF trovato in: %s\n" "$IN_DIR" >&2
  exit 1
fi

TMP_MERGE="$OUT_DIR/_merge.raw.tsv"
: > "$TMP_MERGE"

printf "[1/3] Scansione di %d file in %s e conteggio dei predicati...\n" "$TOTAL" "$IN_DIR"

idx=0
for f in "${FILES[@]}"; do
  idx=$((idx+1))
  base="$(basename "$f")"
  # Rimuovi tutte le estensioni possibili
  stem="${base%.gz}"; stem="${stem%.nt}"; stem="${stem%.rdf}"; stem="${stem%.n3}"; stem="${stem%.ttl}"
  printf "  - [%d/%d] %s\n" "$idx" "$TOTAL" "$base"

  tmp="$PER_FILE_DIR/$stem.raw.tsv"
  out_tsv="$PER_FILE_DIR/$stem.predicates.freq.tsv"
  out_nd="$PER_FILE_DIR/$stem.predicates.ndjson"

  # Determina il tipo di file e processa di conseguenza
  if [[ "$f" == *.nt.gz ]]; then
    # File N-Triples compresso - usa awk diretto (più veloce)
    gzip -cd -- "$f" | awk '
      NF>=3 && $1 !~ /^#/ && $2 ~ /^</ { c[$2]++ }
      END { for (p in c) printf "%d\t%s\n", c[p], p }
    ' > "$tmp"
  elif [[ "$f" == *.nt ]]; then
    # File N-Triples - usa awk diretto (più veloce)
    awk '
      NF>=3 && $1 !~ /^#/ && $2 ~ /^</ { c[$2]++ }
      END { for (p in c) printf "%d\t%s\n", c[p], p }
    ' "$f" > "$tmp"
  else
    # File RDF/N3/Turtle - usa script Python con rdflib (CORRETTO!)
    python3 "$EXTRACTOR" "$f" 2>/dev/null > "$tmp" || {
      echo "    ⚠️  ERRORE: fallito parsing di $f" >&2
      rm -f "$tmp"
      continue
    }
  fi

  # Verifica che il file tmp non sia vuoto
  if [ ! -s "$tmp" ]; then
    echo "    ⚠️  Nessun predicato estratto da $f" >&2
    rm -f "$tmp"
    continue
  fi

  LC_ALL=C sort -nr -k1,1 "$tmp" > "$out_tsv"
  rm -f "$tmp"

  awk -F'\t' '
    {
      raw=$2; iri=raw; gsub(/[<>]/,"",iri);
      n=split(iri,a,/[\/#]/); ln=a[n];
      printf("{\"iri\":\"%s\",\"usage_count\":%s,\"local_name\":\"%s\"}\n", iri, $1, ln)
    }
  ' "$out_tsv" > "$out_nd"

  printf "      → predicati distinti: %d\n" "$(wc -l < "$out_tsv")"

  cat "$out_tsv" >> "$TMP_MERGE"
done

# merge globale
LC_ALL=C awk -F'\t' '{ agg[$2]+=$1 } END { for (p in agg) printf "%d\t%s\n", agg[p], p }' "$TMP_MERGE" \
| LC_ALL=C sort -nr -k1,1 > "$OUT_DIR/predicates.freq.tsv"
rm -f "$TMP_MERGE"

awk -F'\t' '
  {
    raw=$2; iri=raw; gsub(/[<>]/,"",iri);
    n=split(iri,a,/[\/#]/); ln=a[n];
    printf("{\"iri\":\"%s\",\"usage_count\":%s,\"local_name\":\"%s\"}\n", iri, $1, ln)
  }
' "$OUT_DIR/predicates.freq.tsv" > "$OUT_DIR/predicates.ndjson"

# [2/3] form JSON se c'è jq
FORM_OUT="NDJSON: $OUT_DIR/predicates.ndjson"
if command -v jq >/dev/null 2>&1; then
  printf "[2/3] Costruzione del form JSON (wrapper dell'endpoint)...\n"
  BUILT_AT="$(date -u +%FT%TZ)"
  jq -n \
    --arg id "$ENDPOINT_ID" \
    --arg built "$BUILT_AT" \
    --slurpfile preds "$OUT_DIR/predicates.ndjson" '
    {
      version: "v1",
      endpoints: [
        {
          id: $id,
          url: null,
          built_at: $built,
          triples_sampled: null,
          predicates: $preds
        }
      ]
    }' > "$OUT_DIR/predicates.json"
  FORM_OUT="JSON: $OUT_DIR/predicates.json"
fi

# [3/3] riepilogo
printf "[3/3] Completato. Predicati distinti (merge): %d\n" "$(wc -l < "$OUT_DIR/predicates.freq.tsv")"
printf "   → per-file dir: %s\n" "$PER_FILE_DIR"
printf "   → merge TSV:   %s\n" "$OUT_DIR/predicates.freq.tsv"
printf "   → %s\n" "$FORM_OUT"