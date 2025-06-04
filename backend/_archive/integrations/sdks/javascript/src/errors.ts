/**
 * Exceptions personnalisées pour le SDK JavaScript/TypeScript
 */

/**
 * Exception de base pour toutes les erreurs du SDK
 */
export class MAIntelligenceError extends Error {
  public readonly statusCode?: number;
  public readonly requestId?: string;
  public readonly code?: string;

  constructor(
    message: string,
    statusCode?: number,
    requestId?: string,
    code?: string
  ) {
    super(message);
    this.name = 'MAIntelligenceError';
    this.statusCode = statusCode;
    this.requestId = requestId;
    this.code = code;

    // Maintenir la stack trace native
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, MAIntelligenceError);
    }
  }

  /**
   * Représentation string détaillée de l'erreur
   */
  toString(): string {
    const parts = [this.message];
    
    if (this.statusCode) {
      parts.push(`(HTTP ${this.statusCode})`);
    }
    
    if (this.requestId) {
      parts.push(`[Request ID: ${this.requestId}]`);
    }
    
    return parts.join(' ');
  }

  /**
   * Sérialise l'erreur en objet JSON
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      statusCode: this.statusCode,
      requestId: this.requestId,
      code: this.code,
      stack: this.stack
    };
  }
}

/**
 * Erreur d'authentification
 * 
 * Levée quand:
 * - Clé API invalide ou expirée
 * - Token OAuth2 invalide ou expiré
 * - Permissions insuffisantes
 * 
 * @example
 * ```typescript
 * try {
 *   await client.companies.list();
 * } catch (error) {
 *   if (error instanceof AuthenticationError) {
 *     console.log('Vérifiez vos identifiants');
 *     // Rediriger vers la page de login
 *   }
 * }
 * ```
 */
export class AuthenticationError extends MAIntelligenceError {
  constructor(
    message: string = 'Erreur d\'authentification',
    statusCode?: number,
    requestId?: string
  ) {
    super(message, statusCode, requestId, 'AUTHENTICATION_ERROR');
    this.name = 'AuthenticationError';
  }
}

/**
 * Erreur de limitation de taux
 * 
 * Levée quand la limite de requêtes par période est atteinte.
 * 
 * @example
 * ```typescript
 * try {
 *   await client.companies.list();
 * } catch (error) {
 *   if (error instanceof RateLimitError) {
 *     console.log(`Rate limit atteint, attendez ${error.retryAfter}s`);
 *     setTimeout(() => {
 *       // Réessayer la requête
 *     }, error.retryAfter * 1000);
 *   }
 * }
 * ```
 */
export class RateLimitError extends MAIntelligenceError {
  public readonly retryAfter: number;

  constructor(
    message: string = 'Rate limit atteint',
    retryAfter: number = 60,
    statusCode?: number,
    requestId?: string
  ) {
    super(message, statusCode, requestId, 'RATE_LIMIT_ERROR');
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }

  toString(): string {
    return `${super.toString()} (Réessayez dans ${this.retryAfter}s)`;
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      retryAfter: this.retryAfter
    };
  }
}

/**
 * Erreur de ressource non trouvée
 * 
 * Levée quand:
 * - Entreprise avec ID spécifié n'existe pas
 * - Endpoint inexistant
 * 
 * @example
 * ```typescript
 * try {
 *   const company = await client.companies.get('invalid-id');
 * } catch (error) {
 *   if (error instanceof NotFoundError) {
 *     console.log('Entreprise non trouvée');
 *   }
 * }
 * ```
 */
export class NotFoundError extends MAIntelligenceError {
  constructor(
    message: string = 'Ressource non trouvée',
    statusCode: number = 404,
    requestId?: string
  ) {
    super(message, statusCode, requestId, 'NOT_FOUND_ERROR');
    this.name = 'NotFoundError';
  }
}

/**
 * Détail d'une erreur de validation
 */
export interface ValidationErrorDetail {
  field?: string;
  message: string;
  code?: string;
  loc?: (string | number)[];
}

/**
 * Erreur de validation des données
 * 
 * Levée quand:
 * - Données requises manquantes
 * - Format de données invalide
 * - Contraintes de validation non respectées
 * 
 * @example
 * ```typescript
 * try {
 *   await client.companies.create({
 *     siren: 'invalid',  // SIREN invalide
 *     nom_entreprise: ''  // Nom vide
 *   });
 * } catch (error) {
 *   if (error instanceof ValidationError) {
 *     error.details.forEach(detail => {
 *       console.log(`${detail.field}: ${detail.message}`);
 *     });
 *   }
 * }
 * ```
 */
export class ValidationError extends MAIntelligenceError {
  public readonly details: ValidationErrorDetail[];

  constructor(
    message: string = 'Erreur de validation',
    details: any[] = [],
    statusCode: number = 422,
    requestId?: string
  ) {
    super(message, statusCode, requestId, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.details = details.map(detail => this.normalizeDetail(detail));
  }

  private normalizeDetail(detail: any): ValidationErrorDetail {
    if (typeof detail === 'string') {
      return { message: detail };
    }

    if (typeof detail === 'object' && detail !== null) {
      return {
        field: detail.loc ? detail.loc[detail.loc.length - 1] : detail.field,
        message: detail.msg || detail.message || 'Erreur de validation',
        code: detail.type || detail.code,
        loc: detail.loc
      };
    }

    return { message: 'Erreur de validation inconnue' };
  }

  toString(): string {
    let result = super.toString();
    
    if (this.details.length > 0) {
      const detailMessages = this.details.map(detail => {
        const field = detail.field || 'unknown';
        return `${field}: ${detail.message}`;
      });
      result += ` - ${detailMessages.join('; ')}`;
    }
    
    return result;
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      details: this.details
    };
  }
}

/**
 * Erreur de connexion réseau
 * 
 * Levée quand:
 * - Impossible de se connecter à l'API
 * - Timeout de connexion
 * - Erreurs SSL/TLS
 * 
 * @example
 * ```typescript
 * try {
 *   await client.companies.list();
 * } catch (error) {
 *   if (error instanceof ConnectionError) {
 *     console.log('Problème de connexion réseau');
 *     // Afficher un message d'erreur utilisateur
 *   }
 * }
 * ```
 */
export class ConnectionError extends MAIntelligenceError {
  constructor(
    message: string = 'Erreur de connexion',
    code?: string
  ) {
    super(message, undefined, undefined, code || 'CONNECTION_ERROR');
    this.name = 'ConnectionError';
  }
}

/**
 * Erreur serveur (5xx)
 * 
 * Levée quand:
 * - Erreur interne du serveur
 * - Service temporairement indisponible
 * - Erreur de configuration serveur
 * 
 * @example
 * ```typescript
 * try {
 *   await client.companies.list();
 * } catch (error) {
 *   if (error instanceof ServerError) {
 *     console.log('Erreur serveur, réessayez plus tard');
 *     // Implémenter retry avec backoff exponentiel
 *   }
 * }
 * ```
 */
export class ServerError extends MAIntelligenceError {
  constructor(
    message: string = 'Erreur serveur',
    statusCode?: number,
    requestId?: string
  ) {
    super(message, statusCode, requestId, 'SERVER_ERROR');
    this.name = 'ServerError';
  }
}

/**
 * Erreur de configuration du SDK
 * 
 * Levée quand:
 * - Configuration manquante ou invalide
 * - URL de base malformée
 * - Paramètres de client HTTP invalides
 */
export class ConfigurationError extends MAIntelligenceError {
  constructor(message: string = 'Erreur de configuration') {
    super(message, undefined, undefined, 'CONFIGURATION_ERROR');
    this.name = 'ConfigurationError';
  }
}

/**
 * Erreur de parsing de la réponse API
 * 
 * Levée quand:
 * - Réponse JSON malformée
 * - Structure de données inattendue
 * - Type de données incompatible
 */
export class ResponseParsingError extends MAIntelligenceError {
  constructor(message: string = 'Erreur de parsing de la réponse') {
    super(message, undefined, undefined, 'RESPONSE_PARSING_ERROR');
    this.name = 'ResponseParsingError';
  }
}

/**
 * Alias pour compatibilité
 */
export const APIError = MAIntelligenceError;
export const AuthError = AuthenticationError;
export const RateLimitExceeded = RateLimitError;
export const NotFound = NotFoundError;
export const InvalidData = ValidationError;

/**
 * Crée l'exception appropriée selon le code de statut HTTP
 * 
 * @param statusCode Code de statut HTTP
 * @param responseData Données de la réponse d'erreur
 * @returns Exception appropriée
 */
export function createErrorFromResponse(
  statusCode: number,
  responseData: any
): MAIntelligenceError {
  const errorInfo = responseData?.error || {};
  const message = errorInfo.message || `Erreur HTTP ${statusCode}`;
  const requestId = errorInfo.request_id;

  switch (statusCode) {
    case 401:
      return new AuthenticationError(message, statusCode, requestId);
    
    case 403:
      return new AuthenticationError(
        'Permissions insuffisantes',
        statusCode,
        requestId
      );
    
    case 404:
      return new NotFoundError(message, statusCode, requestId);
    
    case 422:
      const details = responseData?.detail || [];
      return new ValidationError(message, details, statusCode, requestId);
    
    case 429:
      const retryAfter = errorInfo?.rate_limit?.retry_after || 60;
      return new RateLimitError(message, retryAfter, statusCode, requestId);
    
    default:
      if (statusCode >= 500) {
        return new ServerError(message, statusCode, requestId);
      }
      return new MAIntelligenceError(message, statusCode, requestId);
  }
}

/**
 * Type guard pour vérifier si une erreur est une MAIntelligenceError
 */
export function isMAIntelligenceError(error: any): error is MAIntelligenceError {
  return error instanceof MAIntelligenceError;
}

/**
 * Type guards pour les différents types d'erreurs
 */
export const isAuthenticationError = (error: any): error is AuthenticationError =>
  error instanceof AuthenticationError;

export const isRateLimitError = (error: any): error is RateLimitError =>
  error instanceof RateLimitError;

export const isNotFoundError = (error: any): error is NotFoundError =>
  error instanceof NotFoundError;

export const isValidationError = (error: any): error is ValidationError =>
  error instanceof ValidationError;

export const isConnectionError = (error: any): error is ConnectionError =>
  error instanceof ConnectionError;

export const isServerError = (error: any): error is ServerError =>
  error instanceof ServerError;