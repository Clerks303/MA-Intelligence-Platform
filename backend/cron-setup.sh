#!/bin/bash
# Configuration Cron Job pour ML Scoring
# M&A Intelligence Platform

set -e

# Configuration
PROJECT_DIR="/path/to/your/project/backend"  # À MODIFIER
PYTHON_PATH="/usr/bin/python3"               # À MODIFIER
LOG_DIR="/var/log/ma-intelligence"
SCRIPT_PATH="$PROJECT_DIR/scoring.py"

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🕐 Configuration Cron Job - ML Scoring${NC}"

# Vérifications préalables
echo -e "${YELLOW}📋 Vérifications préalables...${NC}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Répertoire projet introuvable: $PROJECT_DIR${NC}"
    echo -e "${YELLOW}💡 Modifiez PROJECT_DIR dans ce script${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}❌ Script scoring.py introuvable: $SCRIPT_PATH${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${RED}❌ Fichier .env introuvable${NC}"
    echo -e "${YELLOW}💡 Créez le fichier .env avec SUPABASE_URL et SUPABASE_KEY${NC}"
    exit 1
fi

# Création du répertoire de logs
echo -e "${YELLOW}📁 Création répertoire logs...${NC}"
sudo mkdir -p "$LOG_DIR"
sudo chown $USER:$USER "$LOG_DIR"

# Test du script
echo -e "${YELLOW}🧪 Test du script scoring...${NC}"
cd "$PROJECT_DIR"
if ! $PYTHON_PATH -c "import scoring; print('✅ Import OK')"; then
    echo -e "${RED}❌ Erreur d'import du script${NC}"
    echo -e "${YELLOW}💡 Vérifiez les dépendances: pip install -r requirements-ml.txt${NC}"
    exit 1
fi

# Configuration des tâches cron
echo -e "${YELLOW}⏰ Configuration des tâches cron...${NC}"

# Backup du crontab existant
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# Création du nouveau crontab
cat > /tmp/ma_intelligence_cron << EOF
# M&A Intelligence Platform - ML Scoring Jobs
# Généré automatiquement le $(date)

# Variables d'environnement
PATH=/usr/local/bin:/usr/bin:/bin
SHELL=/bin/bash

# === TÂCHES PRINCIPALES ===

# Scoring quotidien (2h du matin)
0 2 * * * cd $PROJECT_DIR && $PYTHON_PATH scoring.py --log-level INFO >> $LOG_DIR/daily_scoring.log 2>&1

# Scoring hebdomadaire complet avec ré-entraînement (dimanche 1h)
0 1 * * 0 cd $PROJECT_DIR && $PYTHON_PATH scoring.py --force-retrain --log-level INFO >> $LOG_DIR/weekly_retrain.log 2>&1

# === TÂCHES DE MAINTENANCE ===

# Nettoyage des logs anciens (tous les lundis 3h)
0 3 * * 1 find $LOG_DIR -name "*.log" -mtime +30 -delete

# Sauvegarde des modèles (premier jour du mois 4h)
0 4 1 * * cd $PROJECT_DIR && tar -czf $LOG_DIR/models_backup_\$(date +\%Y\%m\%d).tar.gz *.joblib 2>/dev/null || true

# === MONITORING ===

# Vérification santé quotidienne (8h)
0 8 * * * cd $PROJECT_DIR && $PYTHON_PATH -c "
import os
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

try:
    # Vérifier les scores récents
    result = supabase.table('ml_scores').select('calculated_at').order('calculated_at', desc=True).limit(1).execute()
    if result.data:
        last_score = result.data[0]['calculated_at']
        last_date = datetime.fromisoformat(last_score.replace('Z', '+00:00'))
        if (datetime.now() - last_date.replace(tzinfo=None)).days > 2:
            print('⚠️ ALERTE: Aucun score calculé depuis plus de 2 jours')
    else:
        print('⚠️ ALERTE: Aucun score trouvé en base')
except Exception as e:
    print(f'❌ ERREUR MONITORING: {e}')
" >> $LOG_DIR/health_check.log 2>&1

EOF

# Application du nouveau crontab
crontab /tmp/ma_intelligence_cron

echo -e "${GREEN}✅ Cron jobs configurés avec succès!${NC}"
echo -e "${YELLOW}📋 Tâches configurées:${NC}"
echo -e "  🌙 Scoring quotidien: 2h00"
echo -e "  🔄 Ré-entraînement: Dimanche 1h00"
echo -e "  🧹 Nettoyage logs: Lundi 3h00"
echo -e "  💾 Sauvegarde modèles: 1er du mois 4h00"
echo -e "  🏥 Monitoring santé: 8h00"

# Test immédiat (optionnel)
read -p "🧪 Lancer un test de scoring maintenant? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🚀 Test en cours...${NC}"
    cd "$PROJECT_DIR"
    $PYTHON_PATH scoring.py --batch-size 5 --log-level DEBUG
    echo -e "${GREEN}✅ Test terminé${NC}"
fi

# Instructions finales
echo -e "${YELLOW}📝 Instructions:${NC}"
echo -e "  📊 Voir les logs: tail -f $LOG_DIR/daily_scoring.log"
echo -e "  📋 Voir les tâches: crontab -l"
echo -e "  ❌ Supprimer les tâches: crontab -r"
echo -e "  🔧 Modifier les tâches: crontab -e"

echo -e "${GREEN}🎉 Configuration terminée!${NC}"