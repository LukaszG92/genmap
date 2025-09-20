# genmap (PoC)

Modulo di **Query Translation** per `gen:*`:
- Indici locali (VoID/SD/Sevod/statistiche)
- Retrieval RAG (sparse + opz. dense)
- **Una sola** chiamata LLM con output JSON strutturato
- Riscrittura SPARQL con `SERVICE`/`UNION` + grouping tipo ExclusiveGroup
- Cache mappature e telemetria
