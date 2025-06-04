#!/bin/bash

# Script pour lancer le backend simplifié qui fonctionne

# Aller dans le dossier backend
cd "$(dirname "$0")/backend"

# Activer l'environnement virtuel
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
else
    echo "❌ Environnement virtuel 'venv' non trouvé dans backend/"
    exit 1
fi

# Lancer le serveur Uvicorn avec le backend simplifié
echo "🚀 Démarrage du serveur Uvicorn simplifié..."
uvicorn app.main_simple:app --host 0.0.0.0 --port 8000 --reload