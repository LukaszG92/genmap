#!/usr/bin/env bash
# build_affy_predicates.sh — macOS Bash 3.2 friendly, per-file
set -euo pipefail

IN_DIR="${1:-./dumps/GeoNames}"
OUT_DIR="${2:-./predicates/geoNames}"
ENDPOINT_ID="${3:-geoNames}"

PER_FILE_DIR="$OUT_DIR/files"
mkdir -p "$PER_FILE_DIR"

# prepara lista file con glob, evitando pattern letterali
shopt -s nullglob
FILES=( "$IN_DIR"/*.nt "$IN_DIR"/*.nt.gz )
TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
  printf "Nessun file .nt o .nt.gz trovato in: %s\n" "$IN_DIR" >&2
  exit 1
fi

TMP_MERGE="$OUT_DIR/_merge.raw.tsv"
: > "$TMP_MERGE"

printf "[1/3] Scansione di %d file in %s e conteggio dei predicati...\n" "$TOTAL" "$IN_DIR"

idx=0
for f in "${FILES[@]}"; do
  idx=$((idx+1))
  base="$(basename "$f")"
  stem="${base%.gz}"; stem="${stem%.nt}"
  printf "  - [%d/%d] %s\n" "$idx" "$TOTAL" "$base"

  tmp="$PER_FILE_DIR/$stem.raw.tsv"
  out_tsv="$PER_FILE_DIR/$stem.predicates.freq.tsv"
  out_nd="$PER_FILE_DIR/$stem.predicates.ndjson"

  if [[ "$f" == *.gz ]]; then
    gzip -cd -- "$f"
  else
    cat -- "$f"
  fi | awk '
    NF>=3 && $1 !~ /^#/ && $2 ~ /^</ { c[$2]++ }
    END { for (p in c) printf "%d\t%s\n", c[p], p }
  ' > "$tmp"

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
