# Tests End-to-End - M&A Intelligence Platform

## Vue d'ensemble

Cette suite de tests end-to-end valide l'ensemble du pipeline d'enrichissement M&A de bout en bout, depuis l'ingestion des SIREN jusqu'à l'export des données enrichies.

## Architecture des tests

### 🎯 Objectifs

- **Validation complète** du pipeline d'enrichissement multi-sources
- **Vérification du scoring M&A** avec plages de valeurs attendues
- **Contrôle de l'enrichissement contacts** via Kaspr
- **Test des exports** vers CSV et Airtable
- **Validation de la qualité** globale des données

### 🧪 Tests exécutés

1. **Initialisation Orchestrateur** - Configuration et initialisation des clients
2. **Pipeline Enrichissement Complet** - Enrichissement de 5 SIREN fictifs
3. **Validation Scoring M&A** - Vérification des scores et priorités
4. **Validation Contacts Kaspr** - Contrôle des contacts dirigeants
5. **Export CSV** - Export vers fichiers CSV (MA Analysis + Excel)
6. **Export Airtable Mock** - Validation de la structure Airtable
7. **Validation Qualité Données** - Métriques globales de qualité

## Données de test

### SIREN fictifs utilisés

| SIREN     | Entreprise                        | Profil                    | Score attendu | Priorité |
|-----------|-----------------------------------|---------------------------|---------------|----------|
| 123456001 | Cabinet Audit Excellence Paris    | Grande entreprise leader  | 80-95         | HIGH     |
| 123456002 | Expertise Comptable Rhône-Alpes  | Moyenne entreprise        | 65-80         | MEDIUM   |
| 123456003 | Cabinet Martin & Associés         | Cabinet traditionnel      | 55-75         | MEDIUM   |
| 123456004 | Neo Expertise Digitale            | Start-up innovante        | 35-55         | LOW      |
| 123456005 | Comptabilité Services SARL        | Entreprise en difficulté  | 15-35         | LOW      |

### Mode Mock

- **Tous les appels API sont mockés** - Aucun appel réel aux services externes
- **Données réalistes générées** - Simulation de réponses API cohérentes
- **Contacts Kaspr simulés** - Génération de dirigeants fictifs avec coordonnées
- **Exports testés sans destination réelle** - Validation des formats uniquement

## Exécution

### Lancement rapide

```bash
cd backend/
./run_e2e_tests.sh
```

### Lancement manuel

```bash
cd backend/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 test_e2e_ma_pipeline.py
```

### Prérequis

- Python 3.8+
- Modules requis : `asyncio`, `aiohttp`, `pandas`, `sqlalchemy`
- Aucune clé API nécessaire (mode mock)

## Métriques de validation

### Critères de succès

- **Taux d'enrichissement** ≥ 80% (4/5 SIREN)
- **Précision du scoring** ≥ 60% (scores dans plages attendues)
- **Couverture contacts** ≥ 80% (4/5 entreprises avec contacts)
- **Succès exports** = 100% (CSV + Airtable structure)
- **Complétude données** ≥ 70% (champs clés remplis)

### Score de qualité global

Le score global combine toutes les métriques :
- **≥ 80%** : 🏆 Excellent - Pipeline opérationnel
- **≥ 70%** : ✅ Bon - Pipeline fonctionnel
- **≥ 60%** : ⚠️ Acceptable - Améliorations souhaitables  
- **< 60%** : ❌ Insuffisant - Corrections requises

## Interprétation des résultats

### Rapport de test

Le rapport final inclut :
- **Statistiques globales** : nombre de tests, taux de réussite, durée
- **Détail par test** : statut, durée, métriques clés
- **Erreurs et avertissements** : liste des problèmes détectés
- **Score de qualité** : évaluation globale du pipeline
- **Recommandations** : actions à entreprendre

### Codes de sortie

- **0** : Tous les tests passent - Pipeline validé
- **1** : Échecs détectés - Corrections nécessaires

### Exemple de sortie

```
🧪 TESTS END-TO-END - PIPELINE M&A INTELLIGENCE PLATFORM
================================================================================

🧪 Test: 1/7 - Test 01 Orchestrator Initialization
------------------------------------------------------------
   ✅ PASS - Initialisation Orchestrateur (2.34s)
     • Sources activées: 4
     • Configuration: Mock validée
     • Stats initiales: 0 entreprises traitées

🧪 Test: 2/7 - Test 02 Full Enrichment Pipeline
------------------------------------------------------------
   📊 Enrichissement: Cabinet Audit Excellence Paris (123456001)
   📊 Enrichissement: Expertise Comptable Rhône-Alpes (123456002)
   ...
   ✅ PASS - Pipeline Enrichissement Complet (8.67s)
     • Entreprises traitées: 5
     • Taux de succès: 100.0%
     • Sources utilisées: 4

...

📊 STATISTIQUES GLOBALES:
   • Tests exécutés: 7
   • Tests réussis: 7
   • Tests échoués: 0
   • Taux de réussite: 100.0%
   • Durée totale: 15.42s

🎯 QUALITÉ GLOBALE DU PIPELINE: 87.3%
   Status: 🏆 EXCELLENT

🔧 RECOMMANDATIONS:
   ✅ Tous les tests passent - Pipeline opérationnel
   🚀 Prêt pour l'intégration en production
   📊 Enrichissement multi-sources fonctionnel
   🎯 Scoring M&A opérationnel
   👥 Enrichissement contacts Kaspr fonctionnel
   📤 Exports de données opérationnels

🎉 SUCCÈS: Pipeline M&A validé avec succès!
```

## Dépannage

### Erreurs communes

1. **Import modules manquants**
   ```bash
   pip install aiohttp pandas sqlalchemy
   ```

2. **PYTHONPATH incorrect**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Permissions fichier**
   ```bash
   chmod +x run_e2e_tests.sh
   ```

### Mode debug

Pour plus de détails lors des échecs :

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Personnalisation

- Modifier les **SIREN de test** dans `test_companies_data`
- Ajuster les **critères de validation** dans chaque test
- Configurer les **timeouts** selon l'environnement

## Intégration CI/CD

### GitHub Actions

```yaml
- name: Run E2E Tests
  run: |
    cd backend
    ./run_e2e_tests.sh
```

### Utilisation en local

```bash
# Test avant commit
./run_e2e_tests.sh

# Test avec rapport détaillé
python3 test_e2e_ma_pipeline.py > test_report.log 2>&1
```

## Évolutions futures

- **Tests de charge** : enrichissement de 100+ SIREN
- **Tests API réels** : validation avec vraies clés API (env staging)
- **Tests de régression** : comparaison de résultats entre versions
- **Métriques de performance** : temps d'enrichissement par source
- **Tests de robustesse** : gestion des pannes réseau et timeouts