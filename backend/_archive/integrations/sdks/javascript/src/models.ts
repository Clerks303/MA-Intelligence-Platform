/**
 * Modèles de données pour le SDK JavaScript/TypeScript
 */

import { Company, StatusEnum, CompanyFilters, PaginationMeta } from './types';

/**
 * Classe wrapper pour une entreprise avec méthodes utilitaires
 */
export class CompanyModel implements Company {
  public readonly id: string;
  public readonly siren: string;
  public readonly nom_entreprise: string;
  public readonly forme_juridique?: string;
  public readonly date_creation?: string;
  public readonly adresse?: string;
  public readonly ville?: string;
  public readonly code_postal?: string;
  public readonly email?: string;
  public readonly telephone?: string;
  public readonly numero_tva?: string;
  public readonly chiffre_affaires?: number;
  public readonly resultat?: number;
  public readonly effectif?: number;
  public readonly capital_social?: number;
  public readonly code_naf?: string;
  public readonly libelle_code_naf?: string;
  public readonly dirigeant_principal?: string;
  public readonly statut: StatusEnum;
  public readonly score_prospection?: number;
  public readonly description?: string;
  public readonly created_at?: string;
  public readonly updated_at?: string;
  public readonly dirigeants_json?: Record<string, any>;
  public readonly score_details?: Record<string, any>;
  public readonly activity_logs?: Array<Record<string, any>>;
  public readonly details_complets?: Record<string, any>;

  constructor(data: Company) {
    // Copier toutes les propriétés
    Object.assign(this, data);
    
    // Validation de base
    if (!data.id || !data.siren || !data.nom_entreprise) {
      throw new Error('Données d\'entreprise invalides: id, siren et nom_entreprise requis');
    }
    
    this.id = data.id;
    this.siren = data.siren;
    this.nom_entreprise = data.nom_entreprise;
    this.statut = data.statut || StatusEnum.PROSPECT;
  }

  /**
   * Retourne l'adresse complète formatée
   */
  get adresse_complete(): string {
    const parts = [];
    
    if (this.adresse) parts.push(this.adresse);
    if (this.code_postal) parts.push(this.code_postal);
    if (this.ville) parts.push(this.ville);
    
    return parts.join(', ');
  }

  /**
   * Retourne le chiffre d'affaires formaté
   */
  get chiffre_affaires_formatte(): string {
    if (!this.chiffre_affaires) return 'N/A';
    
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(this.chiffre_affaires);
  }

  /**
   * Indique si l'entreprise a des informations de contact
   */
  get a_contact(): boolean {
    return !!(this.email || this.telephone);
  }

  /**
   * Retourne l'âge de l'entreprise en années
   */
  get age_entreprise(): number | null {
    if (!this.date_creation) return null;
    
    const creation = new Date(this.date_creation);
    const maintenant = new Date();
    
    return Math.floor((maintenant.getTime() - creation.getTime()) / (1000 * 60 * 60 * 24 * 365));
  }

  /**
   * Retourne la couleur associée au statut (pour UI)
   */
  get couleur_statut(): string {
    const couleurs = {
      [StatusEnum.PROSPECT]: '#6B7280',      // Gris
      [StatusEnum.CONTACT]: '#3B82F6',       // Bleu
      [StatusEnum.QUALIFICATION]: '#F59E0B', // Jaune
      [StatusEnum.NEGOCIATION]: '#EF4444',   // Rouge
      [StatusEnum.CLIENT]: '#10B981',        // Vert
      [StatusEnum.PERDU]: '#6B7280'          // Gris foncé
    };
    
    return couleurs[this.statut] || couleurs[StatusEnum.PROSPECT];
  }

  /**
   * Retourne le label français du statut
   */
  get libelle_statut(): string {
    const libelles = {
      [StatusEnum.PROSPECT]: 'Prospect',
      [StatusEnum.CONTACT]: 'Contacté',
      [StatusEnum.QUALIFICATION]: 'En qualification',
      [StatusEnum.NEGOCIATION]: 'En négociation',
      [StatusEnum.CLIENT]: 'Client',
      [StatusEnum.PERDU]: 'Perdu'
    };
    
    return libelles[this.statut] || 'Inconnu';
  }

  /**
   * Indique si l'entreprise est un prospect actif
   */
  get est_prospect_actif(): boolean {
    return [
      StatusEnum.PROSPECT,
      StatusEnum.CONTACT,
      StatusEnum.QUALIFICATION,
      StatusEnum.NEGOCIATION
    ].includes(this.statut);
  }

  /**
   * Retourne le score de prospection formaté
   */
  get score_formatte(): string {
    if (!this.score_prospection) return 'N/A';
    
    return `${this.score_prospection.toFixed(1)}/100`;
  }

  /**
   * Retourne la catégorie de score (Faible, Moyen, Élevé)
   */
  get categorie_score(): 'Faible' | 'Moyen' | 'Élevé' | 'N/A' {
    if (!this.score_prospection) return 'N/A';
    
    if (this.score_prospection < 30) return 'Faible';
    if (this.score_prospection < 70) return 'Moyen';
    return 'Élevé';
  }

  /**
   * Retourne les dirigeants sous forme de liste
   */
  get dirigeants_liste(): string[] {
    if (!this.dirigeants_json) return [];
    
    // Format attendu: { "dirigeants": [{"nom": "...", "fonction": "..."}] }
    if (Array.isArray(this.dirigeants_json.dirigeants)) {
      return this.dirigeants_json.dirigeants.map((d: any) => 
        d.fonction ? `${d.nom} (${d.fonction})` : d.nom
      );
    }
    
    return [];
  }

  /**
   * Convertit en objet plain pour sérialisation
   */
  toJSON(): Company {
    return {
      id: this.id,
      siren: this.siren,
      nom_entreprise: this.nom_entreprise,
      forme_juridique: this.forme_juridique,
      date_creation: this.date_creation,
      adresse: this.adresse,
      ville: this.ville,
      code_postal: this.code_postal,
      email: this.email,
      telephone: this.telephone,
      numero_tva: this.numero_tva,
      chiffre_affaires: this.chiffre_affaires,
      resultat: this.resultat,
      effectif: this.effectif,
      capital_social: this.capital_social,
      code_naf: this.code_naf,
      libelle_code_naf: this.libelle_code_naf,
      dirigeant_principal: this.dirigeant_principal,
      statut: this.statut,
      score_prospection: this.score_prospection,
      description: this.description,
      created_at: this.created_at,
      updated_at: this.updated_at,
      dirigeants_json: this.dirigeants_json,
      score_details: this.score_details,
      activity_logs: this.activity_logs,
      details_complets: this.details_complets
    };
  }

  /**
   * Représentation string pour débogage
   */
  toString(): string {
    return `CompanyModel(${this.siren} - ${this.nom_entreprise})`;
  }

  /**
   * Crée une instance depuis des données brutes
   */
  static fromData(data: Company): CompanyModel {
    return new CompanyModel(data);
  }

  /**
   * Crée plusieurs instances depuis un tableau
   */
  static fromArray(data: Company[]): CompanyModel[] {
    return data.map(item => CompanyModel.fromData(item));
  }
}

/**
 * Classe pour construire des filtres de recherche
 */
export class CompanyFiltersBuilder {
  private filters: CompanyFilters = {};

  /**
   * Recherche textuelle globale
   */
  search(query: string): this {
    this.filters.q = query;
    return this;
  }

  /**
   * Filtre par SIREN exact
   */
  bySiren(siren: string): this {
    this.filters.siren = siren;
    return this;
  }

  /**
   * Filtre par nom d'entreprise
   */
  byName(name: string): this {
    this.filters.nom_entreprise = name;
    return this;
  }

  /**
   * Filtre par ville
   */
  byCity(city: string): this {
    this.filters.ville = city;
    return this;
  }

  /**
   * Filtre par code postal
   */
  byPostalCode(postalCode: string): this {
    this.filters.code_postal = postalCode;
    return this;
  }

  /**
   * Filtre par secteur d'activité
   */
  bySector(sector: string): this {
    this.filters.secteur_activite = sector;
    return this;
  }

  /**
   * Filtre par chiffre d'affaires (plage)
   */
  byRevenue(min?: number, max?: number): this {
    if (min !== undefined) this.filters.ca_min = min;
    if (max !== undefined) this.filters.ca_max = max;
    return this;
  }

  /**
   * Filtre par effectif (plage)
   */
  byEmployees(min?: number, max?: number): this {
    if (min !== undefined) this.filters.effectif_min = min;
    if (max !== undefined) this.filters.effectif_max = max;
    return this;
  }

  /**
   * Filtre par score de prospection minimum
   */
  byScore(minScore: number): this {
    this.filters.score_min = minScore;
    return this;
  }

  /**
   * Filtre par date de création (après)
   */
  createdAfter(date: string | Date): this {
    this.filters.date_creation_after = date instanceof Date ? date.toISOString() : date;
    return this;
  }

  /**
   * Filtre par date de création (avant)
   */
  createdBefore(date: string | Date): this {
    this.filters.date_creation_before = date instanceof Date ? date.toISOString() : date;
    return this;
  }

  /**
   * Filtre les entreprises avec email
   */
  withEmail(hasEmail = true): this {
    this.filters.with_email = hasEmail;
    return this;
  }

  /**
   * Filtre les entreprises avec téléphone
   */
  withPhone(hasPhone = true): this {
    this.filters.with_phone = hasPhone;
    return this;
  }

  /**
   * Définit le tri
   */
  sortBy(field: string, order: 'asc' | 'desc' = 'asc'): this {
    this.filters.sort_by = field;
    this.filters.sort_order = order;
    return this;
  }

  /**
   * Retourne les filtres construits
   */
  build(): CompanyFilters {
    return { ...this.filters };
  }

  /**
   * Remet à zéro les filtres
   */
  reset(): this {
    this.filters = {};
    return this;
  }
}

/**
 * Classe wrapper pour les réponses paginées avec méthodes utilitaires
 */
export class PaginatedResponseModel<T> {
  public readonly data: T[];
  public readonly pagination: PaginationMeta;
  public readonly success: boolean;
  public readonly message: string;
  public readonly timestamp?: string;

  constructor(response: {
    data: T[];
    pagination: PaginationMeta;
    success?: boolean;
    message?: string;
    timestamp?: string;
  }) {
    this.data = response.data;
    this.pagination = response.pagination;
    this.success = response.success ?? true;
    this.message = response.message ?? '';
    this.timestamp = response.timestamp;
  }

  /**
   * Itère sur les éléments
   */
  *[Symbol.iterator]() {
    for (const item of this.data) {
      yield item;
    }
  }

  /**
   * Nombre d'éléments dans la page actuelle
   */
  get length(): number {
    return this.data.length;
  }

  /**
   * Premier élément
   */
  get first(): T | undefined {
    return this.data[0];
  }

  /**
   * Dernier élément
   */
  get last(): T | undefined {
    return this.data[this.data.length - 1];
  }

  /**
   * Indique si la réponse est vide
   */
  get isEmpty(): boolean {
    return this.data.length === 0;
  }

  /**
   * Map sur les éléments
   */
  map<U>(fn: (item: T, index: number) => U): U[] {
    return this.data.map(fn);
  }

  /**
   * Filter sur les éléments
   */
  filter(fn: (item: T, index: number) => boolean): T[] {
    return this.data.filter(fn);
  }

  /**
   * Find sur les éléments
   */
  find(fn: (item: T, index: number) => boolean): T | undefined {
    return this.data.find(fn);
  }

  /**
   * Accès par index
   */
  at(index: number): T | undefined {
    return this.data[index];
  }

  /**
   * Slice des éléments
   */
  slice(start?: number, end?: number): T[] {
    return this.data.slice(start, end);
  }

  /**
   * Convertit les entreprises en modèles si applicable
   */
  toCompanyModels(): CompanyModel[] {
    if (this.data.length === 0) return [];
    
    // Vérifier si c'est des données d'entreprise
    const firstItem = this.data[0] as any;
    if (firstItem && typeof firstItem === 'object' && 'siren' in firstItem) {
      return this.data.map(item => new CompanyModel(item as Company));
    }
    
    return [];
  }

  /**
   * Informations de pagination formatées
   */
  get paginationInfo(): string {
    const { page, size, total, total_pages } = this.pagination;
    const start = (page - 1) * size + 1;
    const end = Math.min(page * size, total);
    
    return `${start}-${end} sur ${total} (page ${page}/${total_pages})`;
  }
}

/**
 * Utilitaires pour la manipulation des données
 */
export class DataUtils {
  /**
   * Formate un montant en euros
   */
  static formatCurrency(amount: number | null | undefined): string {
    if (!amount) return 'N/A';
    
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  }

  /**
   * Formate un nombre avec séparateurs
   */
  static formatNumber(num: number | null | undefined): string {
    if (!num) return 'N/A';
    
    return new Intl.NumberFormat('fr-FR').format(num);
  }

  /**
   * Formate une date
   */
  static formatDate(date: string | Date | null | undefined): string {
    if (!date) return 'N/A';
    
    const d = typeof date === 'string' ? new Date(date) : date;
    
    return new Intl.DateTimeFormat('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }).format(d);
  }

  /**
   * Formate une date courte
   */
  static formatDateShort(date: string | Date | null | undefined): string {
    if (!date) return 'N/A';
    
    const d = typeof date === 'string' ? new Date(date) : date;
    
    return new Intl.DateTimeFormat('fr-FR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit'
    }).format(d);
  }

  /**
   * Valide un SIREN
   */
  static isValidSiren(siren: string): boolean {
    if (!siren || siren.length !== 9) return false;
    return /^\d{9}$/.test(siren);
  }

  /**
   * Valide un SIRET
   */
  static isValidSiret(siret: string): boolean {
    if (!siret || siret.length !== 14) return false;
    return /^\d{14}$/.test(siret);
  }

  /**
   * Extrait le SIREN d'un SIRET
   */
  static sirenFromSiret(siret: string): string | null {
    if (!this.isValidSiret(siret)) return null;
    return siret.substring(0, 9);
  }
}