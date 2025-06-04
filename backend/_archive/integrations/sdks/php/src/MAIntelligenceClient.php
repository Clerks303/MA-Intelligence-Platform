<?php

namespace MAIntelligence;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\RequestException;
use GuzzleHttp\Exception\ClientException;
use GuzzleHttp\Exception\ServerException;
use MAIntelligence\Exceptions\MAIntelligenceException;
use MAIntelligence\Exceptions\AuthenticationException;
use MAIntelligence\Exceptions\RateLimitException;
use MAIntelligence\Exceptions\NotFoundException;
use MAIntelligence\Exceptions\ValidationException;
use MAIntelligence\Models\Company;
use MAIntelligence\Models\CompanyFilters;
use MAIntelligence\Models\PaginatedResponse;

/**
 * Client principal pour l'API M&A Intelligence
 * 
 * @example Utilisation de base
 * ```php
 * $client = new MAIntelligenceClient([
 *     'api_key' => 'ak_your_api_key',
 *     'base_url' => 'https://api.ma-intelligence.com'
 * ]);
 * 
 * // Lister les entreprises
 * $companies = $client->companies()->list(['ville' => 'Paris']);
 * 
 * // Recherche avancée
 * $filters = new CompanyFilters();
 * $filters->setQ('comptable')->setVille('Paris')->setWithEmail(true);
 * $results = $client->companies()->search($filters);
 * 
 * // Détails d'une entreprise
 * $company = $client->companies()->get('company-id');
 * ```
 * 
 * @package MAIntelligence
 * @version 1.0.0
 * @author M&A Intelligence Team <sdk@ma-intelligence.com>
 */
class MAIntelligenceClient
{
    private Client $httpClient;
    private string $baseUrl;
    private array $defaultHeaders;
    private int $maxRetries;
    
    private ?CompaniesAPI $companiesAPI = null;
    private ?StatsAPI $statsAPI = null;

    /**
     * Crée une nouvelle instance du client
     * 
     * @param array $config Configuration du client
     * @throws MAIntelligenceException Si la configuration est invalide
     */
    public function __construct(array $config = [])
    {
        // Configuration par défaut
        $defaultConfig = [
            'base_url' => 'https://api.ma-intelligence.com',
            'timeout' => 30,
            'max_retries' => 3,
            'headers' => []
        ];

        $config = array_merge($defaultConfig, $config);

        // Validation de l'authentification
        if (!isset($config['api_key']) && !isset($config['access_token'])) {
            throw new MAIntelligenceException('api_key ou access_token requis');
        }

        $this->baseUrl = rtrim($config['base_url'], '/');
        $this->maxRetries = $config['max_retries'];

        // Configuration des headers d'authentification
        $this->defaultHeaders = [
            'User-Agent' => 'ma-intelligence-sdk-php/1.0.0',
            'Content-Type' => 'application/json',
            'Accept' => 'application/json'
        ];

        if (isset($config['api_key'])) {
            $this->defaultHeaders['X-API-Key'] = $config['api_key'];
        } elseif (isset($config['access_token'])) {
            $this->defaultHeaders['Authorization'] = 'Bearer ' . $config['access_token'];
        }

        // Merger headers personnalisés
        $this->defaultHeaders = array_merge($this->defaultHeaders, $config['headers']);

        // Configuration du client Guzzle
        $this->httpClient = new Client([
            'base_uri' => $this->baseUrl . '/api/v1/',
            'timeout' => $config['timeout'],
            'headers' => $this->defaultHeaders,
            'http_errors' => false // Gérer les erreurs manuellement
        ]);
    }

    /**
     * Interface pour l'API des entreprises
     */
    public function companies(): CompaniesAPI
    {
        if ($this->companiesAPI === null) {
            $this->companiesAPI = new CompaniesAPI($this);
        }
        return $this->companiesAPI;
    }

    /**
     * Interface pour l'API des statistiques
     */
    public function stats(): StatsAPI
    {
        if ($this->statsAPI === null) {
            $this->statsAPI = new StatsAPI($this);
        }
        return $this->statsAPI;
    }

    /**
     * Effectue une requête HTTP
     * 
     * @param string $method Méthode HTTP
     * @param string $endpoint Endpoint de l'API
     * @param array $options Options de la requête
     * @return array Réponse JSON décodée
     * @throws MAIntelligenceException En cas d'erreur
     */
    public function request(string $method, string $endpoint, array $options = []): array
    {
        $attempt = 0;
        
        while ($attempt <= $this->maxRetries) {
            try {
                $response = $this->httpClient->request($method, $endpoint, $options);
                
                return $this->handleResponse($response);
                
            } catch (RequestException $e) {
                $attempt++;
                
                if ($attempt > $this->maxRetries) {
                    throw new MAIntelligenceException(
                        'Erreur réseau: ' . $e->getMessage(),
                        0,
                        $e
                    );
                }
                
                // Attendre avant de réessayer (backoff exponentiel)
                sleep(pow(2, $attempt - 1));
            }
        }
        
        throw new MAIntelligenceException('Nombre maximum de tentatives atteint');
    }

    /**
     * Effectue une requête et retourne le contenu brut
     * 
     * @param string $method Méthode HTTP
     * @param string $endpoint Endpoint de l'API
     * @param array $options Options de la requête
     * @return string Contenu brut de la réponse
     */
    public function requestRaw(string $method, string $endpoint, array $options = []): string
    {
        $response = $this->httpClient->request($method, $endpoint, $options);
        
        if ($response->getStatusCode() >= 400) {
            $this->handleResponse($response); // Déclenche une exception
        }
        
        return $response->getBody()->getContents();
    }

    /**
     * Traite la réponse HTTP
     * 
     * @param \Psr\Http\Message\ResponseInterface $response
     * @return array Données JSON décodées
     * @throws MAIntelligenceException En cas d'erreur
     */
    private function handleResponse($response): array
    {
        $statusCode = $response->getStatusCode();
        $content = $response->getBody()->getContents();
        
        // Gérer le rate limiting
        if ($statusCode === 429) {
            $retryAfter = (int) $response->getHeaderLine('Retry-After') ?: 60;
            throw new RateLimitException(
                'Rate limit atteint',
                $retryAfter,
                $statusCode
            );
        }

        // Gérer l'authentification
        if ($statusCode === 401) {
            throw new AuthenticationException(
                'Authentification échouée. Vérifiez vos identifiants.',
                $statusCode
            );
        }

        // Gérer les permissions
        if ($statusCode === 403) {
            throw new AuthenticationException(
                'Permissions insuffisantes.',
                $statusCode
            );
        }

        // Gérer ressource non trouvée
        if ($statusCode === 404) {
            throw new NotFoundException('Ressource non trouvée.', $statusCode);
        }

        // Gérer erreurs de validation
        if ($statusCode === 422) {
            $errorData = json_decode($content, true);
            $details = $errorData['detail'] ?? [];
            throw new ValidationException(
                'Erreur de validation.',
                $details,
                $statusCode
            );
        }

        // Gérer autres erreurs client/serveur
        if ($statusCode >= 400) {
            $errorData = json_decode($content, true);
            $message = $errorData['error']['message'] ?? "Erreur HTTP $statusCode";
            
            throw new MAIntelligenceException($message, $statusCode);
        }

        // Décoder la réponse JSON
        $data = json_decode($content, true);
        
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new MAIntelligenceException(
                'Réponse invalide (JSON attendu): ' . json_last_error_msg()
            );
        }

        return $data;
    }

    /**
     * Teste la connexion à l'API
     * 
     * @return array Statut de la connexion
     */
    public function testConnection(): array
    {
        try {
            $this->companies()->list(['page' => 1, 'size' => 1]);
            return [
                'status' => 'connected',
                'api_version' => 'v1',
                'message' => 'Connexion réussie'
            ];
        } catch (\Exception $e) {
            return [
                'status' => 'error',
                'message' => $e->getMessage()
            ];
        }
    }
}

/**
 * Interface pour l'API des entreprises
 */
class CompaniesAPI
{
    private MAIntelligenceClient $client;

    public function __construct(MAIntelligenceClient $client)
    {
        $this->client = $client;
    }

    /**
     * Liste les entreprises avec filtres et pagination
     * 
     * @param array $params Paramètres de recherche
     * @return PaginatedResponse
     */
    public function list(array $params = []): PaginatedResponse
    {
        $response = $this->client->request('GET', 'external/companies', [
            'query' => $params
        ]);

        return new PaginatedResponse($response);
    }

    /**
     * Recherche avancée d'entreprises
     * 
     * @param CompanyFilters|array $filters Filtres de recherche
     * @param int $page Numéro de page
     * @param int $size Taille de page
     * @return PaginatedResponse
     */
    public function search($filters, int $page = 1, int $size = 50): PaginatedResponse
    {
        if ($filters instanceof CompanyFilters) {
            $filterData = $filters->toArray();
        } else {
            $filterData = $filters;
        }

        $response = $this->client->request('POST', 'external/companies/search', [
            'json' => $filterData,
            'query' => ['page' => $page, 'size' => $size]
        ]);

        return new PaginatedResponse($response);
    }

    /**
     * Récupère les détails d'une entreprise
     * 
     * @param string $companyId ID de l'entreprise
     * @param bool $includeLogs Inclure les logs d'activité
     * @return Company
     */
    public function get(string $companyId, bool $includeLogs = false): Company
    {
        $params = $includeLogs ? ['include_logs' => 'true'] : [];
        
        $response = $this->client->request('GET', "external/companies/$companyId", [
            'query' => $params
        ]);

        return new Company($response['data']);
    }

    /**
     * Crée une nouvelle entreprise
     * 
     * @param array $companyData Données de l'entreprise
     * @return Company
     */
    public function create(array $companyData): Company
    {
        $response = $this->client->request('POST', 'external/companies', [
            'json' => $companyData
        ]);

        return new Company($response['data']);
    }

    /**
     * Met à jour une entreprise
     * 
     * @param string $companyId ID de l'entreprise
     * @param array $updateData Données à mettre à jour
     * @return Company
     */
    public function update(string $companyId, array $updateData): Company
    {
        $response = $this->client->request('PUT', "external/companies/$companyId", [
            'json' => $updateData
        ]);

        return new Company($response['data']);
    }

    /**
     * Supprime une entreprise
     * 
     * @param string $companyId ID de l'entreprise
     * @return void
     */
    public function delete(string $companyId): void
    {
        $this->client->request('DELETE', "external/companies/$companyId");
    }

    /**
     * Exporte les entreprises en CSV
     * 
     * @param array $options Options d'export
     * @return string Contenu CSV
     */
    public function exportCsv(array $options = []): string
    {
        $params = ['format' => 'csv'];
        
        if (isset($options['filters'])) {
            $params['filters'] = json_encode($options['filters']);
        }

        return $this->client->requestRaw('GET', 'external/export/companies', [
            'query' => $params
        ]);
    }

    /**
     * Import en lot d'entreprises
     * 
     * @param array $importData Données d'import
     * @return array Résultat de l'import
     */
    public function bulkImport(array $importData): array
    {
        $response = $this->client->request('POST', 'external/import/companies', [
            'json' => $importData
        ]);

        return $response['data'];
    }
}

/**
 * Interface pour l'API des statistiques
 */
class StatsAPI
{
    private MAIntelligenceClient $client;

    public function __construct(MAIntelligenceClient $client)
    {
        $this->client = $client;
    }

    /**
     * Récupère les statistiques globales
     * 
     * @param array $params Paramètres des statistiques
     * @return array Statistiques
     */
    public function getGlobal(array $params = []): array
    {
        $response = $this->client->request('GET', 'external/stats', [
            'query' => $params
        ]);

        return $response['data'];
    }
}