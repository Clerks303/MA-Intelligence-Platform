# M&A Intelligence Platform - ML Scoring Makefile
# Commandes pour orchestrer le scoring ML

.PHONY: help install score-all score-recent score-debug score-companies score-test train-models status logs clean

# Configuration
PYTHON := python3
VENV := venv
SCRIPT := scoring.py
LOG_FILE := scoring.log

# Couleurs pour l'affichage
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Affiche cette aide
	@echo "$(GREEN)M&A Intelligence Platform - ML Scoring$(NC)"
	@echo "$(YELLOW)Commandes disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Installe les dépendances ML
	@echo "$(YELLOW)📦 Installation des dépendances ML...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-ml.txt
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

check-env: ## Vérifie la configuration
	@echo "$(YELLOW)🔍 Vérification de l'environnement...$(NC)"
	@if [ -z "$$SUPABASE_URL" ]; then echo "$(RED)❌ SUPABASE_URL manquant$(NC)"; exit 1; fi
	@if [ -z "$$SUPABASE_KEY" ]; then echo "$(RED)❌ SUPABASE_KEY manquant$(NC)"; exit 1; fi
	@echo "$(GREEN)✅ Configuration OK$(NC)"

score-all: check-env ## Lance le scoring pour toutes les entreprises
	@echo "$(YELLOW)🚀 Démarrage scoring complet...$(NC)"
	$(PYTHON) $(SCRIPT) --log-level INFO
	@echo "$(GREEN)✅ Scoring terminé$(NC)"

score-recent: check-env ## Score uniquement les entreprises récentes (derniers 30 jours)
	@echo "$(YELLOW)📅 Scoring des entreprises récentes...$(NC)"
	$(PYTHON) $(SCRIPT) --batch-size 500 --log-level INFO
	@echo "$(GREEN)✅ Scoring récent terminé$(NC)"

score-debug: check-env ## Lance le scoring en mode debug
	@echo "$(YELLOW)🐛 Scoring en mode debug...$(NC)"
	$(PYTHON) $(SCRIPT) --batch-size 10 --log-level DEBUG

score-companies: check-env ## Score des entreprises spécifiques (usage: make score-companies IDS="1,2,3")
	@if [ -z "$(IDS)" ]; then echo "$(RED)❌ Usage: make score-companies IDS=\"1,2,3\"$(NC)"; exit 1; fi
	@echo "$(YELLOW)🎯 Scoring entreprises: $(IDS)$(NC)"
	$(PYTHON) $(SCRIPT) --company-ids $(IDS) --log-level INFO

score-test: check-env ## Lance un scoring de test sur 5 entreprises
	@echo "$(YELLOW)🧪 Test scoring (5 entreprises)...$(NC)"
	$(PYTHON) $(SCRIPT) --batch-size 5 --log-level DEBUG

train-models: check-env ## Force le ré-entraînement des modèles
	@echo "$(YELLOW)🧠 Ré-entraînement des modèles ML...$(NC)"
	$(PYTHON) $(SCRIPT) --force-retrain --batch-size 100 --log-level INFO
	@echo "$(GREEN)✅ Modèles ré-entraînés$(NC)"

status: ## Affiche le statut des derniers scorings
	@echo "$(YELLOW)📊 Statut des scorings...$(NC)"
	@if [ -f $(LOG_FILE) ]; then \
		echo "$(GREEN)Dernières lignes du log:$(NC)"; \
		tail -n 10 $(LOG_FILE); \
	else \
		echo "$(RED)❌ Aucun log trouvé$(NC)"; \
	fi

logs: ## Affiche les logs en temps réel
	@echo "$(YELLOW)📋 Logs en temps réel (Ctrl+C pour arrêter)...$(NC)"
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "$(RED)❌ Aucun log trouvé$(NC)"; \
	fi

clean: ## Nettoie les fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage...$(NC)"
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-models: ## Supprime les modèles sauvegardés (force ré-entraînement)
	@echo "$(YELLOW)🗑️ Suppression des modèles...$(NC)"
	rm -f *.joblib
	@echo "$(GREEN)✅ Modèles supprimés$(NC)"

backup-models: ## Sauvegarde les modèles avec timestamp
	@echo "$(YELLOW)💾 Sauvegarde des modèles...$(NC)"
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p backups; \
	if ls *.joblib 1> /dev/null 2>&1; then \
		tar -czf backups/models_$$timestamp.tar.gz *.joblib; \
		echo "$(GREEN)✅ Modèles sauvegardés: backups/models_$$timestamp.tar.gz$(NC)"; \
	else \
		echo "$(RED)❌ Aucun modèle à sauvegarder$(NC)"; \
	fi

monitor: ## Lance un monitoring continu (refresh toutes les 30s)
	@echo "$(YELLOW)👁️ Monitoring continu (Ctrl+C pour arrêter)...$(NC)"
	@while true; do \
		clear; \
		echo "$(GREEN)=== M&A Intelligence - Monitoring ML ===$(NC)"; \
		echo "$(YELLOW)Timestamp: $$(date)$(NC)"; \
		echo ""; \
		if [ -f $(LOG_FILE) ]; then \
			echo "$(GREEN)Dernières activités:$(NC)"; \
			tail -n 5 $(LOG_FILE) | sed 's/^/  /'; \
		fi; \
		echo ""; \
		echo "$(GREEN)Modèles disponibles:$(NC)"; \
		ls -la *.joblib 2>/dev/null | sed 's/^/  /' || echo "  $(RED)Aucun modèle trouvé$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Prochain refresh dans 30s...$(NC)"; \
		sleep 30; \
	done

# Commandes de déploiement
deploy-cron: ## Configure un cron job quotidien
	@echo "$(YELLOW)⏰ Configuration cron job...$(NC)"
	@cron_entry="0 2 * * * cd $$(pwd) && make score-all >> /var/log/ma-intelligence-scoring.log 2>&1"; \
	(crontab -l 2>/dev/null | grep -v "ma-intelligence-scoring"; echo "$$cron_entry") | crontab -
	@echo "$(GREEN)✅ Cron job configuré (tous les jours à 2h)$(NC)"

remove-cron: ## Supprime le cron job
	@echo "$(YELLOW)🗑️ Suppression cron job...$(NC)"
	@crontab -l 2>/dev/null | grep -v "ma-intelligence-scoring" | crontab -
	@echo "$(GREEN)✅ Cron job supprimé$(NC)"

# Commandes d'environnement
setup-env: ## Configure l'environnement complet
	@echo "$(YELLOW)🔧 Configuration environnement...$(NC)"
	cp .env.example .env 2>/dev/null || echo "$(YELLOW)⚠️ Créez le fichier .env manuellement$(NC)"
	$(MAKE) install
	@echo "$(GREEN)✅ Environnement configuré$(NC)"

# Commandes d'analyse
analyze-scores: ## Analyse les scores en base
	@echo "$(YELLOW)📊 Analyse des scores...$(NC)"
	$(PYTHON) -c "
import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

try:
    result = supabase.table('ml_scores').select('*').execute()
    if result.data:
        df = pd.DataFrame(result.data)
        print(f'📈 Total scores: {len(df)}')
        print(f'📊 Score composite moyen: {df[\"score_composite\"].mean():.1f}')
        print(f'✅ Confiance moyenne: {df[\"confidence\"].mean():.1f}%')
        print(f'📅 Dernier calcul: {df[\"calculated_at\"].max()}')
    else:
        print('❌ Aucun score trouvé')
except Exception as e:
    print(f'❌ Erreur: {e}')
"

# Règle par défaut
.DEFAULT_GOAL := help