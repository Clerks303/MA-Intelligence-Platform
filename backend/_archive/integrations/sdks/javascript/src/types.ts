/**
 * Types TypeScript pour le SDK M&A Intelligence
 */

/**
 * Configuration du client SDK
 */
export interface ClientConfig {
  /** URL de base de l'API */
  baseUrl?: string;
  /** Clé API pour l'authentification */
  apiKey?: string;
  /** Token OAuth2 pour l'authentification */
  accessToken?: string;
  /** Timeout des requêtes en millisecondes */
  timeout?: number;
  /** Nombre maximum de tentatives en cas d'erreur */
  maxRetries?: number;
  /** Headers personnalisés */
  headers?: Record<string, string>;
}

/**
 * Statuts possibles d'une entreprise
 */
export enum StatusEnum {
  PROSPECT = 'prospect',
  CONTACT = 'contact',
  QUALIFICATION = 'qualification',
  NEGOCIATION = 'negociation',
  CLIENT = 'client',
  PERDU = 'perdu'
}

/**
 * Modèle de données d'une entreprise
 */
export interface Company {
  /** Identifiant unique */
  id: string;
  /** Numéro SIREN */
  siren: string;
  /** Nom de l'entreprise */
  nom_entreprise: string;
  
  // Informations légales
  forme_juridique?: string;
  date_creation?: string;
  numero_tva?: string;
  capital_social?: number;
  
  // Adresse
  adresse?: string;
  ville?: string;
  code_postal?: string;
  
  // Contact
  email?: string;
  telephone?: string;
  
  // Activité
  code_naf?: string;
  libelle_code_naf?: string;
  
  // Finances
  chiffre_affaires?: number;
  resultat?: number;
  effectif?: number;
  
  // Management
  dirigeant_principal?: string;
  dirigeants_json?: Record<string, any>;
  
  // Prospection
  statut: StatusEnum;
  score_prospection?: number;
  score_details?: Record<string, any>;
  description?: string;
  
  // Métadonnées
  created_at?: string;
  updated_at?: string;
  activity_logs?: Array<Record<string, any>>;
  details_complets?: Record<string, any>;
}

/**
 * Filtres pour la recherche d'entreprises
 */
export interface CompanyFilters {
  // Recherche textuelle
  q?: string;
  
  // Filtres exacts
  siren?: string;
  nom_entreprise?: string;
  ville?: string;
  code_postal?: string;
  secteur_activite?: string;
  
  // Filtres numériques
  ca_min?: number;
  ca_max?: number;
  effectif_min?: number;
  effectif_max?: number;
  score_min?: number;
  
  // Filtres de dates
  date_creation_after?: string;
  date_creation_before?: string;
  
  // Filtres booléens
  with_email?: boolean;
  with_phone?: boolean;
  
  // Tri
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

/**
 * Paramètres de pagination
 */
export interface PaginationParams {
  /** Numéro de page (commence à 1) */
  page?: number;
  /** Taille de page (max 1000) */
  size?: number;
}

/**
 * Paramètres pour lister les entreprises
 */
export interface ListCompaniesParams extends PaginationParams {
  // Recherche
  q?: string;
  
  // Filtres
  siren?: string;
  ville?: string;
  secteur?: string;
  ca_min?: number;
  ca_max?: number;
  effectif_min?: number;
  effectif_max?: number;
  
  // Tri
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  
  // Options
  include_details?: boolean;
}

/**
 * Paramètres pour la recherche avancée
 */
export interface SearchCompaniesParams extends PaginationParams {
  filters: CompanyFilters;
}

/**
 * Données pour créer une entreprise
 */
export interface CreateCompanyData {
  siren: string;
  nom_entreprise: string;
  forme_juridique?: string;
  date_creation?: string;
  adresse?: string;
  ville?: string;
  code_postal?: string;
  email?: string;
  telephone?: string;
  numero_tva?: string;
  chiffre_affaires?: number;
  resultat?: number;
  effectif?: number;
  capital_social?: number;
  code_naf?: string;
  libelle_code_naf?: string;
  dirigeant_principal?: string;
  statut?: StatusEnum;
  score_prospection?: number;
  description?: string;
}

/**
 * Données pour mettre à jour une entreprise
 */
export interface UpdateCompanyData {
  nom_entreprise?: string;
  email?: string;
  telephone?: string;
  adresse?: string;
  ville?: string;
  code_postal?: string;
  dirigeant_principal?: string;
  chiffre_affaires?: number;
  statut?: StatusEnum;
  effectif?: number;
  capital_social?: number;
  description?: string;
}

/**
 * Réponse API standardisée
 */
export interface APIResponse<T = any> {
  success: boolean;
  data: T;
  message: string;
  timestamp: string;
  request_id?: string;
}

/**
 * Métadonnées de pagination
 */
export interface PaginationMeta {
  page: number;
  size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/**
 * Réponse paginée
 */
export interface PaginatedResponse<T = any> extends APIResponse<T[]> {
  pagination: PaginationMeta;
}

/**
 * Statistiques globales
 */
export interface GlobalStats {
  total: number;
  ca_moyen: number;
  ca_total: number;
  effectif_moyen: number;
  avec_email: number;
  avec_telephone: number;
  taux_email: number;
  taux_telephone: number;
  par_statut: Record<string, number>;
  generated_at: string;
  trends?: {
    period: string;
    daily_growth: number;
    monthly_growth: number;
    note: string;
  };
}

/**
 * Paramètres pour les statistiques
 */
export interface StatsParams {
  include_trends?: boolean;
  date_from?: string;
  date_to?: string;
}

/**
 * Options d'export
 */
export interface ExportOptions {
  format?: 'csv' | 'json' | 'excel';
  filters?: Record<string, any>;
  filename?: string;
}

/**
 * Données pour import en lot
 */
export interface BulkImportData {
  operation: 'create' | 'update' | 'delete';
  data: Array<Record<string, any>>;
  options?: Record<string, any>;
}

/**
 * Résultat d'import en lot
 */
export interface BulkImportResult {
  import_id: string;
  status: 'processing' | 'completed' | 'failed';
  total_items: number;
  processed_items?: number;
  success_count?: number;
  error_count?: number;
  errors?: Array<{
    item: Record<string, any>;
    error: string;
  }>;
}

/**
 * Configuration d'axios interceptée
 */
export interface AxiosConfig {
  baseURL: string;
  timeout: number;
  headers: Record<string, string>;
  maxRedirects: number;
  validateStatus: (status: number) => boolean;
}

/**
 * Informations de rate limiting
 */
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_time: string;
  retry_after?: number;
}

/**
 * Statut de connexion
 */
export interface ConnectionStatus {
  status: 'connected' | 'error';
  api_version?: string;
  message: string;
}