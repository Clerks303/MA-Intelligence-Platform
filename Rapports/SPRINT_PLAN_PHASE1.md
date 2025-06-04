# Sprint Plan - Phase 1 Fondations
## M&A Intelligence Platform - Optimisation (4 semaines)

---

## 🎯 Objectifs Phase 1

**Vision**: Établir les fondations performance, sécurité et observabilité pour supporter la montée en charge

**Objectifs mesurables**:
- Réduire temps de réponse API de 80% (2-5s → <500ms)
- Diminuer les appels API externes de 80% via cache
- Augmenter disponibilité à 99.5%
- Paralléliser enrichissement par lots (3 → 50 concurrent)

---

## 📅 Planning général

| Semaine | Focus | Livrables clés |
|---------|-------|----------------|
| S1 | **Performance BDD + Cache** | PostgreSQL optimisé, Redis opérationnel |
| S2 | **Monitoring + Observabilité** | Prometheus/Grafana, alerting |
| S3 | **Sécurité enterprise** | 2FA, rate limiting, audit logs |
| S4 | **Queues asynchrones** | Celery opérationnel, enrichissement async |

---

## 🗓️ SEMAINE 1 - Performance & Cache
**Objectif**: Optimiser la couche données et implémenter le cache Redis

### 📋 Backlog Sprint 1

#### 🗄️ **US-001: Audit et optimisation PostgreSQL**
**Effort**: 12h | **Priorité**: CRITIQUE

**Description**: Analyser et optimiser les performances de la base de données PostgreSQL pour supporter la montée en charge.

**Tâches détaillées**:
- **T1.1** - Audit requêtes lentes (2h)
  - Activer `pg_stat_statements`
  - Identifier top 10 requêtes les plus lentes
  - Analyser avec `EXPLAIN ANALYZE`
- **T1.2** - Création index composites (4h)
  - Index pour recherche M&A par score/CA
  - Index géographique ville/code postal
  - Index sectoriel code NAF
- **T1.3** - Configuration connection pooling (3h)
  - Configuration PgBouncer ou SQLAlchemy pool
  - Optimisation paramètres PostgreSQL
  - Tests charge connection pool
- **T1.4** - Partitioning tables (3h)
  - Partitioning par date `last_scraped_at`
  - Migration données existantes
  - Tests performances partitions

**Prérequis**:
- Accès admin PostgreSQL
- Backup complet BDD avant modifications
- Environnement de test isolé

**Livrables**:
- `/backend/sql/indexes_optimization.sql`
- `/backend/sql/partitioning_setup.sql`
- `/docs/database_optimization_report.md`
- Configuration PostgreSQL optimisée

**Critères de succès**:
- [ ] Requêtes de recherche <200ms (vs 2-5s)
- [ ] Top 10 requêtes lentes optimisées
- [ ] Connection pool stable 100+ connexions
- [ ] Tests de charge validés

---

#### 🚀 **US-002: Implémentation cache Redis multi-niveaux**
**Effort**: 16h | **Priorité**: CRITIQUE

**Description**: Mettre en place un système de cache Redis sophistiqué pour réduire drastiquement les appels API externes.

**Tâches détaillées**:
- **T2.1** - Setup Redis Cluster (3h)
  - Installation Redis 6+ avec clustering
  - Configuration haute disponibilité
  - Tests failover automatique
- **T2.2** - Module cache distribué (5h)
  - Classe `DistributedCache` avec TTL adaptatif
  - Génération clés déterministes
  - Pattern cache-aside avec fallback
- **T2.3** - Intégration scrapers (4h)
  - Cache Pappers API (TTL 24h)
  - Cache Kaspr contacts (TTL 6h)
  - Cache enrichissement complet (TTL 2h)
- **T2.4** - Cache scoring M&A (2h)
  - Cache résultats scoring (TTL 1h)
  - Invalidation intelligente si données changent
- **T2.5** - Métriques et monitoring (2h)
  - Hit ratio par type de cache
  - Métriques Redis (mémoire, connections)
  - Dashboards cache performance

**Prérequis**:
- Serveur Redis dédié ou Docker
- Accès modification code scrapers
- Bibliothèque `redis-py` installée

**Livrables**:
- `/backend/app/core/cache.py` - Module cache principal
- `/backend/app/scrapers/cached_*.py` - Scrapers avec cache
- `/docker-compose.yml` - Redis service ajouté
- `/docs/cache_strategy.md` - Documentation stratégie

**Critères de succès**:
- [ ] Cache hit ratio >80% après 24h
- [ ] Réduction 80% appels API Pappers/Kaspr
- [ ] Temps réponse enrichissement <500ms
- [ ] Redis stable 1GB+ données cachées

---

#### 📊 **US-003: Métriques de performance**
**Effort**: 6h | **Priorité**: HAUTE

**Description**: Implémenter la collecte de métriques détaillées pour mesurer l'impact des optimisations.

**Tâches détaillées**:
- **T3.1** - Métriques API timing (2h)
  - Middleware FastAPI pour timing requests
  - Métriques par endpoint
  - Percentiles P50, P90, P99
- **T3.2** - Métriques cache (2h)
  - Hit/miss ratio par type
  - Latence cache operations
  - Taille cache par clé pattern
- **T3.3** - Métriques BDD (2h)
  - Slow queries tracking
  - Connection pool utilization
  - Query performance evolution

**Prérequis**:
- Accès modification middleware FastAPI
- Bibliothèque `prometheus-client`

**Livrables**:
- `/backend/app/core/metrics.py` - Collecteur métriques
- `/backend/app/middleware/performance.py` - Middleware timing
- Métriques baseline documentées

**Critères de succès**:
- [ ] Métriques collectées sur tous endpoints
- [ ] Baseline performance établie
- [ ] Alertes automatiques si régression

---

### 📈 **Critères de succès Semaine 1**
- [ ] **Performance**: Requêtes BDD 5x plus rapides
- [ ] **Cache**: Hit ratio >80%, appels API externes -80%
- [ ] **Métriques**: Monitoring complet opérationnel
- [ ] **Stabilité**: Zéro régression fonctionnelle

---

## 🗓️ SEMAINE 2 - Monitoring & Observabilité
**Objectif**: Mise en place monitoring complet et observabilité pour la production

### 📋 Backlog Sprint 2

#### 📊 **US-004: Setup Prometheus + Grafana**
**Effort**: 10h | **Priorité**: CRITIQUE

**Description**: Déployer une stack de monitoring complète avec collecte métriques et dashboards opérationnels.

**Tâches détaillées**:
- **T4.1** - Installation Prometheus (3h)
  - Déploiement Prometheus server
  - Configuration scraping endpoints
  - Règles d'agrégation métriques
- **T4.2** - Setup Grafana (3h)
  - Installation Grafana + datasource Prometheus
  - Import dashboards FastAPI/Redis/PostgreSQL
  - Configuration utilisateurs et permissions
- **T4.3** - Métriques custom business (4h)
  - Métriques enrichissement (taux succès, durée)
  - Métriques scoring (distribution scores)
  - Métriques utilisateurs (connexions, actions)

**Prérequis**:
- Infrastructure Docker ou K8s
- Ports réseau ouverts (9090, 3000)
- Accès modification app pour exposition métriques

**Livrables**:
- `/monitoring/prometheus.yml` - Config Prometheus
- `/monitoring/grafana/dashboards/` - Dashboards JSON
- `/backend/app/core/prometheus_metrics.py` - Métriques custom
- Documentation accès monitoring

**Critères de succès**:
- [ ] Prometheus collecte métriques app
- [ ] Dashboards Grafana opérationnels
- [ ] Métriques business visibles temps réel
- [ ] Formation équipe accès monitoring

---

#### 🚨 **US-005: Alerting intelligent**
**Effort**: 8h | **Priorité**: HAUTE

**Description**: Configurer un système d'alertes proactives pour détecter les problèmes avant impact utilisateurs.

**Tâches détaillées**:
- **T5.1** - Règles d'alerting Prometheus (4h)
  - Seuils API response time >1s
  - Cache hit ratio <70%
  - Erreurs rate >5%
  - Disponibilité services <99%
- **T5.2** - Intégration notifications (2h)
  - Slack webhook pour alertes critiques
  - Email pour alertes warning
  - PagerDuty pour alertes production
- **T5.3** - Runbooks automatisation (2h)
  - Procédures réponse incident
  - Scripts diagnostic automatique
  - Escalation matrix

**Prérequis**:
- Accès Slack workspace
- Comptes email alerting
- Access serveurs pour runbooks

**Livrables**:
- `/monitoring/alerts/rules.yml` - Règles Prometheus
- `/monitoring/alerts/runbooks.md` - Procédures incident
- `/scripts/diagnosis/` - Scripts automatiques
- Matrix escalation incidents

**Critères de succès**:
- [ ] Alertes déclenchées correctement sur tests
- [ ] Notifications reçues dans 30s
- [ ] Runbooks testés et validés
- [ ] Équipe formée sur procédures

---

#### 📝 **US-006: Structured logging**
**Effort**: 6h | **Priorité**: MOYENNE

**Description**: Améliorer le logging applicatif avec logs structurés pour faciliter debugging et monitoring.

**Tâches détaillées**:
- **T6.1** - Migration vers structlog (3h)
  - Configuration structlog JSON
  - Remplacement logs existants
  - Contexte automatique (user_id, request_id)
- **T6.2** - Logs business events (2h)
  - Logs enrichissement success/failure
  - Logs export/scoring events
  - Logs authentification/autorisation
- **T6.3** - Log aggregation (1h)
  - Centralisation logs avec Loki ou ELK
  - Queries et dashboards logs
  - Rétention policy

**Prérequis**:
- Bibliothèque `structlog` installée
- Infrastructure log aggregation

**Livrables**:
- `/backend/app/core/logging.py` - Config structlog
- Logs JSON structurés dans toute l'app
- Dashboards logs Grafana/Kibana

**Critères de succès**:
- [ ] Tous logs en format JSON structuré
- [ ] Recherche logs efficace par contexte
- [ ] Corrélation logs avec métriques

---

### 📈 **Critères de succès Semaine 2**
- [ ] **Observabilité**: Monitoring complet opérationnel
- [ ] **Alerting**: Détection proactive problèmes
- [ ] **Debugging**: Logs structurés facilite investigation
- [ ] **MTTR**: Temps résolution incident réduit 50%

---

## 🗓️ SEMAINE 3 - Sécurité Enterprise
**Objectif**: Renforcer la sécurité pour un environnement de production enterprise

### 📋 Backlog Sprint 3

#### 🛡️ **US-007: Rate limiting granulaire**
**Effort**: 8h | **Priorité**: CRITIQUE

**Description**: Implémenter un système de rate limiting sophistiqué pour protéger contre les abus et garantir la stabilité.

**Tâches détaillées**:
- **T7.1** - Rate limiter Redis (3h)
  - Configuration slowapi avec Redis backend
  - Limits différenciés par type d'opération
  - Sliding window algorithm
- **T7.2** - Protection par endpoint (3h)
  - Enrichissement: 100/hour par user
  - Export: 10/hour par user
  - Login: 5/minute par IP
  - Upload: 5/hour par user
- **T7.3** - Rate limiting intelligent (2h)
  - Limits dynamiques selon plan utilisateur
  - Burst allowance pour pics légitimes
  - Grace period pour nouveaux utilisateurs

**Prérequis**:
- Redis opérationnel (Semaine 1)
- Bibliothèque `slowapi` installée
- Définition plans utilisateurs

**Livrables**:
- `/backend/app/core/rate_limiting.py` - Module rate limiting
- `/backend/app/middleware/rate_limiter.py` - Middleware FastAPI
- Configuration limits par plan utilisateur
- Tests charge rate limiting

**Critères de succès**:
- [ ] Rate limiting actif tous endpoints sensibles
- [ ] Protection effective contre brute force
- [ ] Utilisateurs légitimes non impactés
- [ ] Métriques violations collectées

---

#### 🔐 **US-008: Authentification 2FA**
**Effort**: 10h | **Priorité**: HAUTE

**Description**: Ajouter l'authentification à deux facteurs pour renforcer la sécurité des comptes utilisateurs.

**Tâches détaillées**:
- **T8.1** - Backend 2FA TOTP (4h)
  - Génération secrets TOTP
  - QR codes pour apps mobile
  - Vérification tokens TOTP
  - Backup codes récupération
- **T8.2** - API endpoints 2FA (3h)
  - `/auth/2fa/setup` - Configuration 2FA
  - `/auth/2fa/verify` - Vérification token
  - `/auth/2fa/disable` - Désactivation 2FA
  - `/auth/backup-codes` - Codes récupération
- **T8.3** - Frontend 2FA UI (3h)
  - Formulaire setup 2FA
  - QR code display
  - Input token verification
  - Gestion backup codes

**Prérequis**:
- Bibliothèque `pyotp` pour TOTP
- Bibliothèque `qrcode` pour QR generation
- UI components frontend

**Livrables**:
- `/backend/app/auth/two_factor.py` - Module 2FA
- `/backend/app/api/routes/auth_2fa.py` - Endpoints 2FA
- `/frontend/src/components/auth/TwoFactor.jsx` - UI 2FA
- Documentation utilisateur 2FA

**Critères de succès**:
- [ ] 2FA fonctionnel avec Google Authenticator
- [ ] Backup codes génération/validation
- [ ] UI intuitive et guidée
- [ ] Tests sécurité 2FA validés

---

#### 📋 **US-009: Audit logging complet**
**Effort**: 6h | **Priorité**: MOYENNE

**Description**: Implémenter un système d'audit complet pour traçabilité et compliance.

**Tâches détaillées**:
- **T9.1** - Modèle audit logs (2h)
  - Table `audit_logs` avec champs standardisés
  - Indexation pour recherches efficaces
  - Rétention policy automatique
- **T9.2** - Audit middleware (2h)
  - Logging automatique actions sensibles
  - Capture contexte utilisateur/IP
  - Anonymisation données sensibles
- **T9.3** - Dashboard audit (2h)
  - Vue temps réel activités utilisateurs
  - Filtres par utilisateur/action/date
  - Export logs pour compliance

**Prérequis**:
- Base de données opérationnelle
- Middleware logging (Semaine 2)

**Livrables**:
- `/backend/app/models/audit.py` - Modèle audit
- `/backend/app/middleware/audit.py` - Middleware audit
- Dashboard audit logs Grafana
- Procédures export compliance

**Critères de succès**:
- [ ] Toutes actions sensibles auditées
- [ ] Recherche logs efficace
- [ ] Export compliance fonctionnel
- [ ] Performance impact <5ms par requête

---

### 📈 **Critères de succès Semaine 3**
- [ ] **Sécurité**: Protection contre attaques automatisées
- [ ] **2FA**: Authentification renforcée opérationnelle
- [ ] **Audit**: Traçabilité complète activités
- [ ] **Compliance**: Logs exportables pour certifications

---

## 🗓️ SEMAINE 4 - Queues Asynchrones
**Objectif**: Implémenter le traitement asynchrone pour paralléliser les opérations longues

### 📋 Backlog Sprint 4

#### ⚡ **US-010: Setup Celery + Redis**
**Effort**: 8h | **Priorité**: CRITIQUE

**Description**: Déployer Celery avec Redis comme broker pour traitement asynchrone des tâches longues.

**Tâches détaillées**:
- **T10.1** - Installation Celery (3h)
  - Configuration Celery app
  - Redis comme broker et backend
  - Multiple queues (enrichment, export, scoring)
  - Worker pools configuration
- **T10.2** - Infrastructure queues (2h)
  - Docker services Celery workers
  - Scaling horizontal workers
  - Health checks workers
- **T10.3** - Monitoring Flower (3h)
  - Interface web Flower
  - Métriques tâches en cours
  - Historique success/failure rates
  - Intégration alerting

**Prérequis**:
- Redis opérationnel (Semaine 1)
- Bibliothèque `celery[redis]` installée
- Infrastructure Docker/K8s

**Livrables**:
- `/backend/app/core/celery_app.py` - Config Celery
- `/backend/celery_worker.py` - Worker startup
- `/docker-compose.yml` - Services Celery ajoutés
- Dashboard Flower opérationnel

**Critères de succès**:
- [ ] Celery workers démarrent correctement
- [ ] Redis broker stable sous charge
- [ ] Flower monitoring accessible
- [ ] Auto-scaling workers fonctionnel

---

#### 🔄 **US-011: Migration enrichissement asynchrone**
**Effort**: 10h | **Priorité**: CRITIQUE

**Description**: Migrer l'enrichissement par lots vers traitement asynchrone pour améliorer performance et UX.

**Tâches détaillées**:
- **T11.1** - Tâches Celery enrichissement (4h)
  - Task `enrich_company_async`
  - Task `enrich_batch_async` avec progress tracking
  - Retry policy avec backoff exponentiel
  - Error handling et reporting
- **T11.2** - API endpoints async (3h)
  - `/scraping/enrich-async` - Enrichissement simple
  - `/scraping/batch-async` - Upload CSV + traitement
  - `/tasks/{task_id}/status` - Suivi progression
  - `/tasks/{task_id}/result` - Récupération résultat
- **T11.3** - Frontend task tracking (3h)
  - Composant upload avec progress bar
  - Polling status task en cours
  - Notification fin traitement
  - Gestion erreurs asynchrones

**Prérequis**:
- Celery opérationnel (US-010)
- Orchestrateur enrichissement existant
- UI components progress tracking

**Livrables**:
- `/backend/app/tasks/enrichment.py` - Tâches Celery
- `/backend/app/api/routes/async_scraping.py` - Endpoints async
- `/frontend/src/components/AsyncProcessing.jsx` - UI async
- Tests charge enrichissement parallel

**Critères de succès**:
- [ ] Enrichissement 50 entreprises parallèle
- [ ] Progress tracking temps réel
- [ ] Upload CSV non-bloquant
- [ ] Gestion erreurs robuste

---

#### 📤 **US-012: Exports asynchrones**
**Effort**: 6h | **Priorité**: HAUTE

**Description**: Rendre les exports volumineux asynchrones pour éviter timeouts et améliorer UX.

**Tâches détaillées**:
- **T12.1** - Tâches export async (3h)
  - Task `export_csv_async`
  - Task `export_airtable_async`
  - Task `export_sql_async`
  - Stockage temporaire fichiers
- **T12.2** - Download endpoints (2h)
  - `/exports/{task_id}/download` - Téléchargement fichier
  - `/exports/{task_id}/status` - Statut export
  - Nettoyage automatique fichiers temporaires
- **T12.3** - UI exports async (1h)
  - Boutons export avec feedback async
  - Notifications email fin export (optionnel)
  - Historique exports utilisateur

**Prérequis**:
- Export Manager existant (v2.0)
- Celery opérationnel (US-010)
- Stockage temporaire fichiers

**Livrables**:
- `/backend/app/tasks/exports.py` - Tâches export
- `/backend/app/api/routes/async_exports.py` - Endpoints download
- UI exports améliorée
- Cleanup automatique fichiers

**Critères de succès**:
- [ ] Exports >1000 entreprises asynchrones
- [ ] Téléchargement fichiers sécurisé
- [ ] Nettoyage automatique après 24h
- [ ] Notifications utilisateur

---

### 📈 **Critères de succès Semaine 4**
- [ ] **Parallélisation**: 50+ enrichissements simultanés
- [ ] **UX**: Opérations longues non-bloquantes
- [ ] **Scalabilité**: Workers auto-scaling
- [ ] **Fiabilité**: Retry automatique + monitoring

---

## 📊 Definition of Done - Phase 1

### 🎯 **Critères globaux de succès**

#### **Performance**
- [ ] Temps réponse API <500ms (P90)
- [ ] Cache hit ratio >80%
- [ ] Requêtes BDD optimisées 5x plus rapides
- [ ] 50+ enrichissements parallèles opérationnels

#### **Observabilité**
- [ ] Monitoring Prometheus/Grafana complet
- [ ] Alerting proactif configuré et testé
- [ ] Logs structurés dans toute l'application
- [ ] Métriques business collectées

#### **Sécurité**
- [ ] Rate limiting protège tous endpoints
- [ ] 2FA fonctionnel et testé
- [ ] Audit logs complets et searchables
- [ ] Tests sécurité réalisés et validés

#### **Scalabilité**
- [ ] Queue asynchrone Celery stable
- [ ] Workers auto-scaling opérationnel
- [ ] Traitement batch non-bloquant
- [ ] Infrastructure prête montée en charge

### 🧪 **Tests requis**

#### **Tests de charge**
- [ ] 1000+ requêtes simultanées
- [ ] 100+ enrichissements parallèles
- [ ] Cache stable 10GB+ données
- [ ] BDD stable 1000+ connexions

#### **Tests de sécurité**
- [ ] Rate limiting résiste brute force
- [ ] 2FA résiste attaques connues
- [ ] Audit logs intègres
- [ ] OWASP Top 10 validé

#### **Tests de résilience**
- [ ] Failover Redis automatique
- [ ] Recovery workers après crash
- [ ] Dégradation gracieuse services
- [ ] Backup/restore procédures

### 📚 **Documentation livrée**

- [ ] **Architecture**: Schémas infrastructure mise à jour
- [ ] **Runbooks**: Procédures opérationnelles
- [ ] **Monitoring**: Guide utilisation dashboards
- [ ] **Sécurité**: Procédures 2FA et audit
- [ ] **Déploiement**: Scripts automatisation
- [ ] **Formation**: Guide équipe technique

---

## ⚙️ **Setup et prérequis techniques**

### 🛠️ **Infrastructure requise**

#### **Serveurs/Services**
- **Redis Cluster** (3 nodes minimum)
- **PostgreSQL** optimisé (version 14+)
- **Prometheus** + **Grafana** monitoring
- **Celery Workers** (scaling horizontal)

#### **Développement**
- **Python 3.11+** avec venv
- **Docker** + **Docker Compose**
- **Git** avec branches feature
- **IDE** avec support Python/FastAPI

### 📦 **Dépendances nouvelles**

```txt
# Ajouts requirements.txt
redis==4.5.4
celery[redis]==5.2.7
flower==1.2.0
slowapi==0.1.7
prometheus-client==0.16.0
structlog==23.1.0
pyotp==2.8.0
qrcode==7.4.2
psycopg2-binary==2.9.6
```

### 🔧 **Configuration environnement**

```env
# Ajouts .env
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
FLOWER_PORT=5555
```

### 👥 **Équipe et rôles**

#### **Rôles requis**
- **Tech Lead** (architecture, reviews)
- **DevOps** (infrastructure, monitoring)
- **Dev Backend** (2x, implémentation)
- **Dev Frontend** (1x, UI async)
- **QA** (tests charge, sécurité)

#### **Compétences critiques**
- **Redis/Celery** (expérience queues async)
- **PostgreSQL** (optimisation BDD)
- **Prometheus/Grafana** (monitoring)
- **Sécurité** (2FA, rate limiting)

---

## 🚀 **Next Steps Phase 2**

À l'issue de la Phase 1, l'équipe sera prête pour :

1. **Phase 2 - Qualité** (CI/CD, API versioning)
2. **Phase 3 - Intelligence** (ML scoring)
3. **Phase 4 - Scale** (Microservices)

**Planning suggéré**: Démarrage Phase 2 immédiatement après validation Phase 1 pour maintenir la dynamique d'optimisation.

---

*Ce plan de sprint est conçu pour être **actionnable immédiatement** avec des tâches clairement définies, des efforts réalistes et des critères de succès mesurables. Chaque sprint build sur les précédents pour construire progressivement les fondations solides de votre plateforme M&A Intelligence.*