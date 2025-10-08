# genmap (PoC)

Modulo di **Query Translation** per `gen:*`:
- Indici locali (VoID/SD/Sevod/statistiche)
- Retrieval RAG (sparse + opz. dense)
- **Una sola** chiamata LLM con output JSON strutturato
- Riscrittura SPARQL con `SERVICE`/`UNION` + grouping tipo ExclusiveGroup
- Cache mappature e telemetria

test_indices.py usage: python3 ./src/genmap/index/test_indices.py -q "gen:partOf" --topk 3
