import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { AlertWithIcon } from '../ui/alert';
import { Badge } from '../ui/badge';
import { 
  Settings,
  Euro,
  Star,
  Hash,
  TrendingUp,
  Info
} from 'lucide-react';
import { cn } from '../../lib/utils';

export function ParametersCard({ 
  parameters, 
  onParametersChange, 
  className 
}) {
  const handleChange = (key, value) => {
    onParametersChange({ ...parameters, [key]: value });
  };

  const formatCurrency = (value) => {
    if (!value) return '';
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR',
      maximumFractionDigits: 0
    }).format(value);
  };

  return (
    <Card className={cn("transition-all duration-300 hover:shadow-lg", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-primary" />
          Paramètres d'enrichissement
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Configurez les critères pour l'enrichissement automatique des données
        </p>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* CA Minimum */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2">
              <Euro className="h-4 w-4 text-green-500" />
              CA minimum
            </label>
            <div className="relative">
              <Input
                type="number"
                placeholder="0"
                value={parameters.min_ca}
                onChange={(e) => handleChange('min_ca', parseInt(e.target.value) || 0)}
                className="pl-8"
              />
              <Euro className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            </div>
            <p className="text-xs text-muted-foreground">
              Minimum: {formatCurrency(parameters.min_ca)}
            </p>
          </div>

          {/* Score Minimum */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2">
              <Star className="h-4 w-4 text-yellow-500" />
              Score minimum
            </label>
            <div className="relative">
              <Input
                type="number"
                placeholder="0"
                min="0"
                max="100"
                value={parameters.min_score}
                onChange={(e) => handleChange('min_score', parseInt(e.target.value) || 0)}
                className="pl-8"
              />
              <Star className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">0%</span>
              <span className="text-muted-foreground">{parameters.min_score}%</span>
              <span className="text-muted-foreground">100%</span>
            </div>
          </div>

          {/* SIREN Spécifique */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2">
              <Hash className="h-4 w-4 text-blue-500" />
              SIREN spécifique
            </label>
            <div className="relative">
              <Input
                placeholder="Laisser vide pour tous"
                value={parameters.siren}
                onChange={(e) => handleChange('siren', e.target.value)}
                className="pl-8 font-mono text-sm"
              />
              <Hash className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            </div>
            <p className="text-xs text-muted-foreground">
              {parameters.siren ? `Ciblé: ${parameters.siren}` : 'Toutes les entreprises'}
            </p>
          </div>
        </div>

        {/* Active Parameters Summary */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Critères actifs
          </h4>
          
          <div className="flex flex-wrap gap-2">
            {parameters.min_ca > 0 && (
              <Badge variant="outline" className="gap-1">
                <Euro className="h-3 w-3" />
                CA ≥ {formatCurrency(parameters.min_ca)}
              </Badge>
            )}
            
            {parameters.min_score > 0 && (
              <Badge variant="outline" className="gap-1">
                <Star className="h-3 w-3" />
                Score ≥ {parameters.min_score}%
              </Badge>
            )}
            
            {parameters.siren && (
              <Badge variant="outline" className="gap-1">
                <Hash className="h-3 w-3" />
                SIREN: {parameters.siren}
              </Badge>
            )}
            
            {parameters.min_ca === 0 && parameters.min_score === 0 && !parameters.siren && (
              <Badge variant="secondary">
                Aucun filtre - toutes les entreprises
              </Badge>
            )}
          </div>
        </div>

        {/* Information Alert */}
        <AlertWithIcon variant="info">
          <div className="text-sm space-y-2">
            <p className="font-medium">Comment ça fonctionne :</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>
                <strong>CA minimum</strong> : Enrichit uniquement les entreprises avec un chiffre d'affaires supérieur au seuil
              </li>
              <li>
                <strong>Score minimum</strong> : Filtre par score de prospection (0-100%)
              </li>
              <li>
                <strong>SIREN spécifique</strong> : Cible une entreprise précise (utile pour les tests)
              </li>
            </ul>
          </div>
        </AlertWithIcon>

        {/* Quick Presets */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium">Préréglages rapides</h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => onParametersChange({ min_ca: 1000000, min_score: 50, siren: '' })}
              className="p-3 text-left border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="font-medium text-sm">PME Standard</div>
              <div className="text-xs text-muted-foreground">CA ≥ 1M€, Score ≥ 50%</div>
            </button>
            
            <button
              onClick={() => onParametersChange({ min_ca: 10000000, min_score: 70, siren: '' })}
              className="p-3 text-left border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="font-medium text-sm">Grandes Entreprises</div>
              <div className="text-xs text-muted-foreground">CA ≥ 10M€, Score ≥ 70%</div>
            </button>
            
            <button
              onClick={() => onParametersChange({ min_ca: 0, min_score: 80, siren: '' })}
              className="p-3 text-left border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="font-medium text-sm">Haute Qualité</div>
              <div className="text-xs text-muted-foreground">Tout CA, Score ≥ 80%</div>
            </button>
            
            <button
              onClick={() => onParametersChange({ min_ca: 0, min_score: 0, siren: '' })}
              className="p-3 text-left border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <div className="font-medium text-sm">Tout Enrichir</div>
              <div className="text-xs text-muted-foreground">Aucun filtre</div>
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}