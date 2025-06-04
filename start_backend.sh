#!/bin/bash

# Aller dans le dossier backend
cd "$(dirname "$0")/backend"
pip install -r requirements-backend.txt

# Activer l’environnement virtuel
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Environnement virtuel 'venv' non trouvé dans backend/"
    exit 1
fi

# Vérifier que backend/__init__.py existe
if [ ! -f "__init__.py" ]; then
    touch __init__.py
    echo "✅ Création de backend/__init__.py"
fi

# Vérifier que backend/api/__init__.py existe
if [ ! -d "api" ]; then
    mkdir api
    echo "✅ Création du dossier backend/api/"
fi
if [ ! -f "api/__init__.py" ]; then
    touch api/__init__.py
    echo "✅ Création de backend/api/__init__.py"
fi

# Revenir à la racine pour lancer uvicorn depuis le bon chemin
cd ..

# Lancer le serveur Uvicorn depuis la racine
echo "🚀 Démarrage du serveur Uvicorn..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
