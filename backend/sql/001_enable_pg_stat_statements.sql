-- Activation de pg_stat_statements pour audit des requêtes
-- Ce script doit être exécuté par un superuser PostgreSQL

-- 1. Ajouter pg_stat_statements aux shared_preload_libraries
-- (nécessite redémarrage PostgreSQL si pas déjà fait)

-- 2. Créer l'extension si elle n'existe pas
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 3. Réinitialiser les statistiques pour une mesure propre
SELECT pg_stat_statements_reset();

-- 4. Vérifier que l'extension fonctionne
SELECT 
    calls,
    total_exec_time,
    mean_exec_time,
    query
FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 5;

-- 5. Configuration recommandée (à ajouter dans postgresql.conf)
/*
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all
pg_stat_statements.track_utility = on
pg_stat_statements.save = on
*/