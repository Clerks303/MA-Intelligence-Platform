#!/bin/bash

# Script de lancement des tests End-to-End pour M&A Intelligence Platform
# Exécute une suite complète de tests avec données mock

echo "🧪 Lancement des Tests End-to-End - M&A Intelligence Platform"
echo "=============================================================="

# Vérification que nous sommes dans le bon répertoire
if [ ! -f "test_e2e_ma_pipeline.py" ]; then
    echo "❌ Erreur: test_e2e_ma_pipeline.py non trouvé"
    echo "   Assurez-vous d'être dans le répertoire backend/"
    exit 1
fi

# Vérification de Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python 3 non trouvé"
    echo "   Installez Python 3 pour continuer"
    exit 1
fi

# Configuration de l'environnement
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export KASPR_API_KEY=""  # Vide pour forcer le mode mock
export AIRTABLE_API_KEY=""  # Vide pour forcer le mode mock

echo "🔧 Configuration:"
echo "   • Mode: MOCK (aucun appel API réel)"
echo "   • SIREN de test: 5 entreprises fictives"
echo "   • Sources: Pappers, Infogreffe, Société.com, Kaspr"
echo ""

# Exécution des tests
echo "🚀 Démarrage des tests..."
python3 test_e2e_ma_pipeline.py

# Récupération du code de sortie
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Tests E2E terminés avec SUCCÈS!"
    echo "   Pipeline M&A validé et opérationnel"
else
    echo "💥 Tests E2E terminés avec des ERREURS"
    echo "   Consultez les logs ci-dessus pour corriger"
fi

echo ""
echo "📋 Fichiers de sortie générés:"
echo "   • Logs détaillés affichés ci-dessus"
echo "   • Fichiers CSV de test dans /tmp/ma_exports/"
echo ""

exit $exit_code