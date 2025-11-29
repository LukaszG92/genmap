#!/bin/bash

# Script per cancellare tutti i file .n3 in una directory
# Utilizzo: ./delete_n3_files.sh [percorso_directory]

# Se non viene specificata una directory, usa la directory corrente
DIR="${1:-.}"

# Verifica che la directory esista
if [ ! -d "$DIR" ]; then
    echo "Errore: La directory '$DIR' non esiste."
    exit 1
fi

# Trova tutti i file .n3 nella directory
N3_FILES=$(find "$DIR" -maxdepth 1 -type f -name "*.n3")

# Conta quanti file .n3 sono presenti
COUNT=$(echo "$N3_FILES" | grep -c "\.n3$" 2>/dev/null || echo "0")

if [ "$COUNT" -eq 0 ] || [ -z "$N3_FILES" ]; then
    echo "Nessun file .n3 trovato in '$DIR'"
    exit 0
fi

# Mostra i file che verranno cancellati
echo "Trovati $COUNT file .n3 in '$DIR':"
echo "$N3_FILES"
echo ""

# Chiedi conferma
read -p "Vuoi cancellare questi file? (s/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    # Cancella i file
    find "$DIR" -maxdepth 1 -type f -name "*.n3" -delete
    echo "File .n3 cancellati con successo!"
else
    echo "Operazione annullata."
fi