/**
 * Client principal du SDK JavaScript/TypeScript pour M&A Intelligence
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import {
  ClientConfig,
  Company,
  CompanyFilters,
  ListCompaniesParams,
  SearchCompaniesParams,
  CreateCompanyData,
  UpdateCompanyData,
  APIResponse,
  PaginatedResponse,
  GlobalStats,
  StatsParams,
  ExportOptions,
  BulkImportData,
  BulkImportResult,
  ConnectionStatus
} from './types';
import {
  MAIntelligenceError,
  AuthenticationError,
  RateLimitError,
  NotFoundError,
  ValidationError,
  ConnectionError,
  ServerError
} from './errors';

/**
 * Interface pour l'API des entreprises
 */
export class CompaniesAPI {
  constructor(private client: MAIntelligenceClient) {}

  /**
   * Liste les entreprises avec filtres et pagination
   * 
   * @example
   * ```typescript
   * const companies = await client.companies.list({
   *   ville: 'Paris',
   *   ca_min: 100000,
   *   page: 1,
   *   size: 50
   * });
   * 
   * console.log(`Trouvé ${companies.pagination.total} entreprises`);
   * companies.data.forEach(company => {
   *   console.log(`${company.nom_entreprise} - ${company.chiffre_affaires}€`);
   * });
   * ```
   */
  async list(params: ListCompaniesParams = {}): Promise<PaginatedResponse<Company>> {
    const response = await this.client.request<PaginatedResponse<Company>>(
      'GET',
      '/external/companies',
      { params }
    );
    return response.data;
  }

  /**
   * Recherche avancée d'entreprises
   * 
   * @example
   * ```typescript
   * const results = await client.companies.search({
   *   filters: {
   *     q: 'comptable',
   *     ville: 'Paris',
   *     ca_min: 50000,
   *     with_email: true
   *   },
   *   page: 1,
   *   size: 100
   * });
   * ```
   */
  async search(params: SearchCompaniesParams): Promise<PaginatedResponse<Company>> {
    const { filters, page = 1, size = 50 } = params;
    
    const response = await this.client.request<PaginatedResponse<Company>>(
      'POST',
      '/external/companies/search',
      {
        data: filters,
        params: { page, size }
      }
    );
    return response.data;
  }

  /**
   * Récupère les détails d'une entreprise
   * 
   * @example
   * ```typescript
   * const company = await client.companies.get('123e4567-e89b-12d3-a456-426614174000', {
   *   include_logs: true
   * });
   * console.log(`${company.nom_entreprise} - SIREN: ${company.siren}`);
   * ```
   */
  async get(
    companyId: string, 
    options: { include_logs?: boolean } = {}
  ): Promise<Company> {
    const response = await this.client.request<APIResponse<Company>>(
      'GET',
      `/external/companies/${companyId}`,
      { params: options }
    );
    return response.data.data;
  }

  /**
   * Crée une nouvelle entreprise
   * 
   * @example
   * ```typescript
   * const newCompany = await client.companies.create({
   *   siren: '123456789',
   *   nom_entreprise: 'SARL Exemple',
   *   ville: 'Paris',
   *   email: 'contact@exemple.fr'
   * });
   * ```
   */
  async create(companyData: CreateCompanyData): Promise<Company> {
    const response = await this.client.request<APIResponse<Company>>(
      'POST',
      '/external/companies',
      { data: companyData }
    );
    return response.data.data;
  }

  /**
   * Met à jour une entreprise existante
   * 
   * @example
   * ```typescript
   * const updated = await client.companies.update('company-id', {
   *   email: 'nouveau@email.fr',
   *   telephone: '0123456789'
   * });
   * ```
   */
  async update(companyId: string, updateData: UpdateCompanyData): Promise<Company> {
    const response = await this.client.request<APIResponse<Company>>(
      'PUT',
      `/external/companies/${companyId}`,
      { data: updateData }
    );
    return response.data.data;
  }

  /**
   * Supprime une entreprise
   * 
   * @example
   * ```typescript
   * await client.companies.delete('company-id');
   * ```
   */
  async delete(companyId: string): Promise<void> {
    await this.client.request('DELETE', `/external/companies/${companyId}`);
  }

  /**
   * Exporte les entreprises
   * 
   * @example
   * ```typescript
   * // Export CSV avec filtres
   * const csvData = await client.companies.exportCsv({
   *   format: 'csv',
   *   filters: { ville: 'Paris' }
   * });
   * 
   * // Sauvegarder le fichier (Node.js)
   * require('fs').writeFileSync('companies.csv', csvData);
   * ```
   */
  async exportCsv(options: ExportOptions = {}): Promise<string | ArrayBuffer> {
    const params: any = { format: options.format || 'csv' };
    
    if (options.filters) {
      params.filters = JSON.stringify(options.filters);
    }

    const response = await this.client.request(
      'GET',
      '/external/export/companies',
      { 
        params,
        responseType: 'blob'
      }
    );
    
    return response.data;
  }

  /**
   * Import en lot d'entreprises
   * 
   * @example
   * ```typescript
   * const result = await client.companies.bulkImport({
   *   operation: 'create',
   *   data: [
   *     { siren: '123456789', nom_entreprise: 'Entreprise 1' },
   *     { siren: '987654321', nom_entreprise: 'Entreprise 2' }
   *   ]
   * });
   * 
   * console.log(`Import lancé: ${result.import_id}`);
   * ```
   */
  async bulkImport(importData: BulkImportData): Promise<BulkImportResult> {
    const response = await this.client.request<APIResponse<BulkImportResult>>(
      'POST',
      '/external/import/companies',
      { data: importData }
    );
    return response.data.data;
  }
}

/**
 * Interface pour l'API des statistiques
 */
export class StatsAPI {
  constructor(private client: MAIntelligenceClient) {}

  /**
   * Récupère les statistiques globales
   * 
   * @example
   * ```typescript
   * const stats = await client.stats.getGlobal({
   *   include_trends: true,
   *   date_from: '2024-01-01',
   *   date_to: '2024-12-31'
   * });
   * 
   * console.log(`Total: ${stats.total} entreprises`);
   * console.log(`CA moyen: ${stats.ca_moyen.toFixed(2)}€`);
   * ```
   */
  async getGlobal(params: StatsParams = {}): Promise<GlobalStats> {
    const response = await this.client.request<APIResponse<GlobalStats>>(
      'GET',
      '/external/stats',
      { params }
    );
    return response.data.data;
  }
}

/**
 * Client principal pour l'API M&A Intelligence
 * 
 * @example Utilisation de base
 * ```typescript
 * import { MAIntelligenceClient } from '@ma-intelligence/sdk';
 * 
 * const client = new MAIntelligenceClient({
 *   apiKey: 'ak_your_api_key',
 *   baseUrl: 'https://api.ma-intelligence.com'
 * });
 * 
 * // Utilisation
 * const companies = await client.companies.list({ ville: 'Paris' });
 * const stats = await client.stats.getGlobal();
 * ```
 * 
 * @example Avec gestion d'erreurs
 * ```typescript
 * try {
 *   const companies = await client.companies.list();
 * } catch (error) {
 *   if (error instanceof RateLimitError) {
 *     console.log(`Rate limit atteint, réessayez dans ${error.retryAfter}s`);
 *   } else if (error instanceof AuthenticationError) {
 *     console.log('Erreur d\'authentification');
 *   } else {
 *     console.log('Erreur:', error.message);
 *   }
 * }
 * ```
 */
export class MAIntelligenceClient {
  private axios: AxiosInstance;

  /** Interface pour l'API des entreprises */
  public readonly companies: CompaniesAPI;
  
  /** Interface pour l'API des statistiques */
  public readonly stats: StatsAPI;

  /**
   * Crée une nouvelle instance du client
   * 
   * @param config Configuration du client
   */
  constructor(config: ClientConfig) {
    if (!config.apiKey && !config.accessToken) {
      throw new MAIntelligenceError('apiKey ou accessToken requis');
    }

    // Configuration par défaut
    const defaultConfig = {
      baseUrl: 'https://api.ma-intelligence.com',
      timeout: 30000,
      maxRetries: 3,
      headers: {}
    };

    const finalConfig = { ...defaultConfig, ...config };

    // Configuration des headers d'authentification
    const headers: Record<string, string> = {
      'User-Agent': '@ma-intelligence/sdk-js/1.0.0',
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...finalConfig.headers
    };

    if (finalConfig.apiKey) {
      headers['X-API-Key'] = finalConfig.apiKey;
    } else if (finalConfig.accessToken) {
      headers['Authorization'] = `Bearer ${finalConfig.accessToken}`;
    }

    // Configuration d'axios
    this.axios = axios.create({
      baseURL: `${finalConfig.baseUrl.replace(/\/$/, '')}/api/v1`,
      timeout: finalConfig.timeout,
      headers,
      maxRedirects: 5,
      validateStatus: (status) => status < 500 // Gérer les erreurs 4xx manuellement
    });

    // Intercepteur de réponse pour gestion d'erreurs
    this.axios.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        throw this.handleError(error);
      }
    );

    // Initialiser les interfaces API
    this.companies = new CompaniesAPI(this);
    this.stats = new StatsAPI(this);
  }

  /**
   * Effectue une requête HTTP
   * 
   * @internal
   */
  async request<T = any>(
    method: string,
    endpoint: string,
    options: {
      params?: any;
      data?: any;
      headers?: Record<string, string>;
      responseType?: 'json' | 'blob' | 'text';
    } = {}
  ): Promise<AxiosResponse<T>> {
    const config = {
      method: method.toLowerCase(),
      url: endpoint,
      ...options
    };

    return this.axios.request<T>(config);
  }

  /**
   * Teste la connexion à l'API
   * 
   * @example
   * ```typescript
   * const status = await client.testConnection();
   * console.log(`Connexion: ${status.status}`);
   * ```
   */
  async testConnection(): Promise<ConnectionStatus> {
    try {
      await this.companies.list({ page: 1, size: 1 });
      return {
        status: 'connected',
        api_version: 'v1',
        message: 'Connexion réussie'
      };
    } catch (error) {
      return {
        status: 'error',
        message: error instanceof Error ? error.message : 'Erreur inconnue'
      };
    }
  }

  /**
   * Gère les erreurs de requête
   * 
   * @private
   */
  private handleError(error: AxiosError): Error {
    // Erreurs de connexion
    if (!error.response) {
      return new ConnectionError(
        `Erreur de connexion: ${error.message}`,
        error.code
      );
    }

    const { status, data } = error.response;
    const errorData = data as any;
    const errorInfo = errorData?.error || {};
    const message = errorInfo.message || `Erreur HTTP ${status}`;
    const requestId = errorInfo.request_id;

    // Rate limiting
    if (status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      return new RateLimitError(
        message,
        parseInt(retryAfter) || 60,
        status,
        requestId
      );
    }

    // Authentification
    if (status === 401) {
      return new AuthenticationError(
        'Authentification échouée. Vérifiez vos identifiants.',
        status,
        requestId
      );
    }

    // Permissions
    if (status === 403) {
      return new AuthenticationError(
        'Permissions insuffisantes.',
        status,
        requestId
      );
    }

    // Ressource non trouvée
    if (status === 404) {
      return new NotFoundError(message, status, requestId);
    }

    // Validation
    if (status === 422) {
      const details = errorData?.detail || [];
      return new ValidationError(message, details, status, requestId);
    }

    // Erreurs serveur
    if (status >= 500) {
      return new ServerError(message, status, requestId);
    }

    // Autres erreurs
    return new MAIntelligenceError(message, status, requestId);
  }
}