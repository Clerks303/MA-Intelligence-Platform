<?php

namespace MAIntelligence\Models;

/**
 * Modèle de données pour une entreprise
 * 
 * @package MAIntelligence\Models
 */
class Company
{
    public string $id;
    public string $siren;
    public string $nom_entreprise;
    public ?string $forme_juridique = null;
    public ?string $date_creation = null;
    public ?string $adresse = null;
    public ?string $ville = null;
    public ?string $code_postal = null;
    public ?string $email = null;
    public ?string $telephone = null;
    public ?string $numero_tva = null;
    public ?float $chiffre_affaires = null;
    public ?float $resultat = null;
    public ?int $effectif = null;
    public ?float $capital_social = null;
    public ?string $code_naf = null;
    public ?string $libelle_code_naf = null;
    public ?string $dirigeant_principal = null;
    public string $statut = 'prospect';
    public ?float $score_prospection = null;
    public ?string $description = null;
    public ?string $created_at = null;
    public ?string $updated_at = null;
    public ?array $dirigeants_json = null;
    public ?array $score_details = null;
    public ?array $activity_logs = null;
    public ?array $details_complets = null;

    /**
     * Constructeur
     * 
     * @param array $data Données de l'entreprise
     */
    public function __construct(array $data = [])
    {
        foreach ($data as $key => $value) {
            if (property_exists($this, $key)) {
                $this->$key = $value;
            }
        }

        // Validation de base
        if (empty($this->id) || empty($this->siren) || empty($this->nom_entreprise)) {
            throw new \InvalidArgumentException(
                'Données d\'entreprise invalides: id, siren et nom_entreprise requis'
            );
        }
    }

    /**
     * Retourne l'adresse complète formatée
     */
    public function getAdresseComplete(): string
    {
        $parts = array_filter([
            $this->adresse,
            $this->code_postal,
            $this->ville
        ]);

        return implode(', ', $parts);
    }

    /**
     * Retourne le chiffre d'affaires formaté
     */
    public function getChiffreAffairesFormatte(): string
    {
        if ($this->chiffre_affaires === null) {
            return 'N/A';
        }

        return number_format($this->chiffre_affaires, 0, ',', ' ') . ' €';
    }

    /**
     * Indique si l'entreprise a des informations de contact
     */
    public function aContact(): bool
    {
        return !empty($this->email) || !empty($this->telephone);
    }

    /**
     * Retourne l'âge de l'entreprise en années
     */
    public function getAgeEntreprise(): ?int
    {
        if (!$this->date_creation) {
            return null;
        }

        $creation = new \DateTime($this->date_creation);
        $maintenant = new \DateTime();

        return (int) $creation->diff($maintenant)->format('%y');
    }

    /**
     * Retourne la couleur associée au statut
     */
    public function getCouleurStatut(): string
    {
        $couleurs = [
            'prospect' => '#6B7280',
            'contact' => '#3B82F6',
            'qualification' => '#F59E0B',
            'negociation' => '#EF4444',
            'client' => '#10B981',
            'perdu' => '#6B7280'
        ];

        return $couleurs[$this->statut] ?? $couleurs['prospect'];
    }

    /**
     * Retourne le label français du statut
     */
    public function getLibelleStatut(): string
    {
        $libelles = [
            'prospect' => 'Prospect',
            'contact' => 'Contacté',
            'qualification' => 'En qualification',
            'negociation' => 'En négociation',
            'client' => 'Client',
            'perdu' => 'Perdu'
        ];

        return $libelles[$this->statut] ?? 'Inconnu';
    }

    /**
     * Indique si l'entreprise est un prospect actif
     */
    public function estProspectActif(): bool
    {
        return in_array($this->statut, [
            'prospect',
            'contact',
            'qualification',
            'negociation'
        ]);
    }

    /**
     * Retourne le score de prospection formaté
     */
    public function getScoreFormatte(): string
    {
        if ($this->score_prospection === null) {
            return 'N/A';
        }

        return number_format($this->score_prospection, 1) . '/100';
    }

    /**
     * Retourne la catégorie de score
     */
    public function getCategorieScore(): string
    {
        if ($this->score_prospection === null) {
            return 'N/A';
        }

        if ($this->score_prospection < 30) {
            return 'Faible';
        }

        if ($this->score_prospection < 70) {
            return 'Moyen';
        }

        return 'Élevé';
    }

    /**
     * Retourne les dirigeants sous forme de liste
     */
    public function getDirigeantsListe(): array
    {
        if (!$this->dirigeants_json || !isset($this->dirigeants_json['dirigeants'])) {
            return [];
        }

        $dirigeants = [];
        foreach ($this->dirigeants_json['dirigeants'] as $dirigeant) {
            if (isset($dirigeant['fonction'])) {
                $dirigeants[] = $dirigeant['nom'] . ' (' . $dirigeant['fonction'] . ')';
            } else {
                $dirigeants[] = $dirigeant['nom'];
            }
        }

        return $dirigeants;
    }

    /**
     * Convertit en tableau pour sérialisation
     */
    public function toArray(): array
    {
        $data = [];
        
        foreach (get_object_vars($this) as $property => $value) {
            $data[$property] = $value;
        }

        return $data;
    }

    /**
     * Représentation string pour débogage
     */
    public function __toString(): string
    {
        return "Company({$this->siren} - {$this->nom_entreprise})";
    }

    /**
     * Crée une instance depuis des données
     */
    public static function fromArray(array $data): self
    {
        return new self($data);
    }

    /**
     * Crée plusieurs instances depuis un tableau
     */
    public static function fromArrayCollection(array $data): array
    {
        return array_map(fn($item) => self::fromArray($item), $data);
    }
}