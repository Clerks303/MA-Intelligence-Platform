-- Index composites optimisés pour M&A Intelligence Platform
-- Basé sur les patterns de requêtes identifiés dans l'application

-- =============================================
-- INDEX POUR RECHERCHE M&A (priorité haute)
-- =============================================

-- Index composite pour recherche par score M&A + CA + effectif
-- Utilisé dans: /api/v1/companies/search avec filtres M&A
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_ma_search 
ON companies (ma_score DESC, chiffre_affaires DESC, effectif DESC) 
WHERE ma_score IS NOT NULL 
AND ma_score >= 50 
AND statut != 'refuse'
AND last_scraped_at > CURRENT_DATE - INTERVAL '90 days';

-- Index pour recherche par critères financiers
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_financial_criteria 
ON companies (chiffre_affaires DESC, resultat DESC, effectif DESC)
WHERE chiffre_affaires IS NOT NULL 
AND chiffre_affaires >= 1000000;

-- Index pour scoring rapide (lookup par SIREN)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_scoring_lookup
ON companies (siren, ma_score, last_scraped_at)
WHERE ma_score IS NOT NULL;

-- =============================================
-- INDEX GÉOGRAPHIQUES
-- =============================================

-- Index géographique pour recherche par zone
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_location_score 
ON companies (ville, code_postal, ma_score DESC) 
WHERE statut != 'refuse' 
AND ma_score IS NOT NULL;

-- Index pour recherche par département (2 premiers caractères code postal)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_department
ON companies (LEFT(code_postal, 2), ma_score DESC)
WHERE code_postal IS NOT NULL 
AND ma_score >= 40;

-- =============================================
-- INDEX SECTORIELS
-- =============================================

-- Index GIN pour recherche par secteur d'activité
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_sector_analysis 
ON companies USING GIN (code_naf gin_trgm_ops, secteur_activite gin_trgm_ops) 
WHERE last_scraped_at > CURRENT_DATE - INTERVAL '30 days';

-- Index pour analyse sectorielle avec score
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_naf_score
ON companies (LEFT(code_naf, 2), ma_score DESC, chiffre_affaires DESC)
WHERE code_naf IS NOT NULL;

-- =============================================
-- INDEX TEMPORELS ET ENRICHISSEMENT
-- =============================================

-- Index pour suivi enrichissement récent
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_enrichment_status
ON companies (last_scraped_at DESC, statut, ma_score)
WHERE last_scraped_at IS NOT NULL;

-- Index pour entreprises à re-enrichir
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_stale_data
ON companies (last_scraped_at ASC, siren)
WHERE last_scraped_at < CURRENT_DATE - INTERVAL '30 days'
OR last_scraped_at IS NULL;

-- =============================================
-- INDEX POUR ANALYTICS ET STATS
-- =============================================

-- Index pour dashboard analytics rapide
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_analytics
ON companies (created_at, statut, ma_score)
WHERE created_at >= CURRENT_DATE - INTERVAL '1 year';

-- Index pour calculs de percentiles
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_percentiles
ON companies (ma_score, chiffre_affaires)
WHERE ma_score IS NOT NULL 
AND chiffre_affaires IS NOT NULL;

-- =============================================
-- INDEX POUR CONTACTS (table company_contacts)
-- =============================================

-- Index pour récupération contacts par entreprise
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contacts_company_priority
ON company_contacts (company_id, est_dirigeant DESC, confidence_score DESC)
WHERE est_dirigeant = true;

-- Index pour recherche contacts avec email
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contacts_with_email
ON company_contacts (company_id, email_professionnel)
WHERE email_professionnel IS NOT NULL 
AND email_professionnel != '';

-- =============================================
-- INDEX POUR AUDIT LOGS (future table)
-- =============================================

-- Index pour audit logs (sera utilisé en semaine 3)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_lookup
ON audit_logs (user_id, created_at DESC, action)
WHERE created_at >= CURRENT_DATE - INTERVAL '1 year';

-- =============================================
-- STATISTIQUES ET MAINTENANCE
-- =============================================

-- Activer autovacuum agressif sur tables principales
ALTER TABLE companies SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05,
    autovacuum_vacuum_cost_limit = 1000
);

ALTER TABLE company_contacts SET (
    autovacuum_vacuum_scale_factor = 0.2,
    autovacuum_analyze_scale_factor = 0.1
);

-- Forcer analyse des statistiques
ANALYZE companies;
ANALYZE company_contacts;

-- =============================================
-- REQUÊTES DE VÉRIFICATION
-- =============================================

-- Vérifier que les index sont utilisés
-- (à exécuter après création des index)

/*
-- Test recherche M&A
EXPLAIN (ANALYZE, BUFFERS) 
SELECT siren, nom_entreprise, ma_score, chiffre_affaires 
FROM companies 
WHERE ma_score >= 70 
AND chiffre_affaires >= 5000000 
ORDER BY ma_score DESC, chiffre_affaires DESC 
LIMIT 50;

-- Test recherche géographique  
EXPLAIN (ANALYZE, BUFFERS)
SELECT siren, nom_entreprise, ville, ma_score
FROM companies 
WHERE ville = 'Paris' 
AND ma_score >= 60
ORDER BY ma_score DESC
LIMIT 20;

-- Test recherche sectorielle
EXPLAIN (ANALYZE, BUFFERS)
SELECT LEFT(code_naf, 2) as secteur, COUNT(*), AVG(ma_score)
FROM companies 
WHERE code_naf IS NOT NULL 
AND ma_score IS NOT NULL
GROUP BY LEFT(code_naf, 2)
ORDER BY AVG(ma_score) DESC;
*/