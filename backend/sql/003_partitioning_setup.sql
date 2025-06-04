-- Mise en place du partitioning par date pour optimiser les performances
-- Ce script configure le partitioning sur la table companies par last_scraped_at

-- =============================================
-- CONFIGURATION PARTITIONING TABLE COMPANIES
-- =============================================

-- 1. Créer la table companies partitionnée (si migration complète)
-- NOTE: Cette section est commentée car elle nécessite une migration complète
/*
-- Renommer la table existante
ALTER TABLE companies RENAME TO companies_old;

-- Créer la nouvelle table partitionnée
CREATE TABLE companies (
    LIKE companies_old INCLUDING ALL
) PARTITION BY RANGE (last_scraped_at);

-- Créer les partitions pour les 12 derniers mois
CREATE TABLE companies_2024_q1 PARTITION OF companies
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE companies_2024_q2 PARTITION OF companies
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE companies_2024_q3 PARTITION OF companies
FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE companies_2024_q4 PARTITION OF companies
FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

CREATE TABLE companies_2025_q1 PARTITION OF companies
FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Partition pour données sans date (NULL last_scraped_at)
CREATE TABLE companies_unscraped PARTITION OF companies
FOR VALUES FROM (MINVALUE) TO ('2020-01-01');

-- Migrer les données
INSERT INTO companies SELECT * FROM companies_old;

-- Vérifier la migration
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename LIKE 'companies%';

-- Supprimer l'ancienne table (ATTENTION: faire un backup avant!)
-- DROP TABLE companies_old;
*/

-- =============================================
-- SOLUTION ALTERNATIVE: INDEX PARTITIONNEL
-- =============================================

-- Pour éviter la migration complète, on utilise des index partiels
-- qui simulent le comportement du partitioning

-- Index pour données récentes (derniers 30 jours)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_recent_data
ON companies (last_scraped_at DESC, siren)
WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '30 days';

-- Index pour données du dernier trimestre
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_quarter_data  
ON companies (last_scraped_at DESC, ma_score DESC)
WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '90 days'
AND ma_score IS NOT NULL;

-- Index pour données de l'année courante
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_year_data
ON companies (last_scraped_at DESC, created_at DESC)
WHERE last_scraped_at >= date_trunc('year', CURRENT_DATE);

-- Index pour données anciennes (archivage)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_archive_data
ON companies (last_scraped_at ASC, siren)
WHERE last_scraped_at < CURRENT_DATE - INTERVAL '1 year'
OR last_scraped_at IS NULL;

-- =============================================
-- PARTITIONING TABLE AUDIT_LOGS (future)
-- =============================================

-- Préparer le partitioning pour les logs d'audit
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Partitions par mois pour les 6 derniers mois
CREATE TABLE IF NOT EXISTS audit_logs_2024_01 PARTITION OF audit_logs
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_02 PARTITION OF audit_logs
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_03 PARTITION OF audit_logs
FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_04 PARTITION OF audit_logs
FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_05 PARTITION OF audit_logs
FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

CREATE TABLE IF NOT EXISTS audit_logs_2024_06 PARTITION OF audit_logs
FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

-- Index sur les partitions
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_time 
ON audit_logs (user_id, created_at DESC, action);

CREATE INDEX IF NOT EXISTS idx_audit_logs_resource_time
ON audit_logs (resource, created_at DESC);

-- =============================================
-- FONCTION DE MAINTENANCE AUTOMATIQUE
-- =============================================

-- Fonction pour créer automatiquement les nouvelles partitions
CREATE OR REPLACE FUNCTION create_monthly_audit_partition(target_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
    create_sql TEXT;
BEGIN
    -- Calculer les dates de début et fin du mois
    start_date := date_trunc('month', target_date)::DATE;
    end_date := (start_date + INTERVAL '1 month')::DATE;
    
    -- Nom de la partition
    partition_name := 'audit_logs_' || to_char(start_date, 'YYYY_MM');
    
    -- SQL de création
    create_sql := format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF audit_logs FOR VALUES FROM (%L) TO (%L)',
        partition_name,
        start_date,
        end_date
    );
    
    -- Exécuter la création
    EXECUTE create_sql;
    
    -- Créer les index sur la nouvelle partition
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_%I_user_time ON %I (user_id, created_at DESC)',
        partition_name, partition_name
    );
    
    RAISE NOTICE 'Partition % créée pour la période % - %', partition_name, start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- Fonction pour nettoyer les anciennes partitions
CREATE OR REPLACE FUNCTION cleanup_old_audit_partitions(retention_months INTEGER DEFAULT 12)
RETURNS VOID AS $$
DECLARE
    partition_record RECORD;
    cutoff_date DATE;
BEGIN
    cutoff_date := (CURRENT_DATE - (retention_months || ' months')::INTERVAL)::DATE;
    
    FOR partition_record IN
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'audit_logs_20%'
        AND tablename < 'audit_logs_' || to_char(cutoff_date, 'YYYY_MM')
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I.%I', partition_record.schemaname, partition_record.tablename);
        RAISE NOTICE 'Partition supprimée: %', partition_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- VUES MATERIALISÉES POUR ANALYTICS
-- =============================================

-- Vue matérialisée pour les statistiques de scraping par mois
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_scraping_stats_monthly AS
SELECT 
    date_trunc('month', last_scraped_at) as month,
    COUNT(*) as companies_scraped,
    COUNT(CASE WHEN ma_score >= 70 THEN 1 END) as high_potential,
    AVG(ma_score) as avg_ma_score,
    COUNT(CASE WHEN details_complets IS NOT NULL THEN 1 END) as with_complete_data
FROM companies 
WHERE last_scraped_at IS NOT NULL
AND last_scraped_at >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY date_trunc('month', last_scraped_at)
ORDER BY month DESC;

-- Index unique pour rafraîchissement concurrentiel
CREATE UNIQUE INDEX IF NOT EXISTS mv_scraping_stats_monthly_unique 
ON mv_scraping_stats_monthly (month);

-- Vue matérialisée pour les top secteurs par score M&A
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_sectors_ma AS
SELECT 
    LEFT(code_naf, 2) as secteur_naf,
    libelle_code_naf,
    COUNT(*) as nb_entreprises,
    AVG(ma_score) as score_moyen,
    AVG(chiffre_affaires) as ca_moyen,
    COUNT(CASE WHEN ma_score >= 80 THEN 1 END) as top_prospects
FROM companies 
WHERE code_naf IS NOT NULL 
AND ma_score IS NOT NULL
AND last_scraped_at >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY LEFT(code_naf, 2), libelle_code_naf
HAVING COUNT(*) >= 5  -- Au moins 5 entreprises par secteur
ORDER BY score_moyen DESC, nb_entreprises DESC;

-- Index pour la vue secteurs
CREATE UNIQUE INDEX IF NOT EXISTS mv_top_sectors_ma_unique 
ON mv_top_sectors_ma (secteur_naf);

-- =============================================
-- PROCÉDURES DE MAINTENANCE AUTOMATIQUE
-- =============================================

-- Rafraîchir les vues matérialisées (à exécuter quotidiennement)
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS VOID AS $$
BEGIN
    -- Rafraîchissement concurrent pour éviter le lock
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_scraping_stats_monthly;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_top_sectors_ma;
    
    RAISE NOTICE 'Vues matérialisées rafraîchies à %', NOW();
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- CONFIGURATION AUTOVACUUM SPÉCIALISÉE
-- =============================================

-- Configuration autovacuum pour la table companies (principale)
ALTER TABLE companies SET (
    autovacuum_vacuum_scale_factor = 0.1,    -- Vacuum plus fréquent
    autovacuum_analyze_scale_factor = 0.05,  -- Analyze plus fréquent
    autovacuum_vacuum_cost_limit = 2000,     -- Plus de ressources pour vacuum
    autovacuum_vacuum_cost_delay = 10        -- Délai réduit
);

-- Configuration pour les contacts (insertion fréquente)
ALTER TABLE company_contacts SET (
    autovacuum_vacuum_scale_factor = 0.2,
    autovacuum_analyze_scale_factor = 0.1,
    autovacuum_vacuum_cost_limit = 1000
);

-- Configuration pour les audit logs (écriture intensive)
ALTER TABLE audit_logs SET (
    autovacuum_vacuum_scale_factor = 0.05,   -- Très fréquent
    autovacuum_analyze_scale_factor = 0.02,  -- Très fréquent
    autovacuum_vacuum_cost_limit = 3000      -- Plus de ressources
);

-- =============================================
-- REQUÊTES DE VALIDATION
-- =============================================

-- Vérifier l'efficacité du partitioning (à exécuter après mise en place)
/*
-- Statistiques par partition
SELECT 
    schemaname,
    tablename,
    n_tup_ins as insertions,
    n_tup_upd as updates,
    n_tup_del as deletions,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables 
WHERE tablename LIKE 'companies%' 
OR tablename LIKE 'audit_logs%'
ORDER BY tablename;

-- Taille des partitions
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename LIKE 'companies%' 
OR tablename LIKE 'audit_logs%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Performance des requêtes typiques
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM companies 
WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '30 days'
AND ma_score >= 70
ORDER BY ma_score DESC
LIMIT 50;
*/

-- =============================================
-- SCRIPT DE VALIDATION POST-INSTALLATION
-- =============================================

-- Forcer l'analyse des statistiques après création des index
ANALYZE companies;
ANALYZE company_contacts;
ANALYZE audit_logs;

-- Validation finale
SELECT 
    'Partitioning setup completed' as status,
    NOW() as executed_at,
    COUNT(*) as total_companies,
    COUNT(CASE WHEN last_scraped_at IS NOT NULL THEN 1 END) as scraped_companies,
    COUNT(CASE WHEN last_scraped_at >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_companies
FROM companies;