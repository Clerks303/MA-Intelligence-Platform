import React from 'react';
import { Card, CardContent } from '../ui/card';
import { Input } from '../ui/input';
import { Select } from '../ui/select';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { 
  Search, 
  Euro, 
  Users, 
  MapPin, 
  Filter,
  X,
  RotateCcw
} from 'lucide-react';
import { cn } from '../../lib/utils';

export function CompanyFilters({ 
  filters, 
  onFiltersChange, 
  cities = [],
  className 
}) {
  const handleFilterChange = (key, value) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const clearFilters = () => {
    onFiltersChange({
      ca_min: '',
      effectif_min: '',
      ville: 'all',
      statut: 'all',
      search: ''
    });
  };

  const activeFiltersCount = Object.values(filters).filter(value => 
    value && value !== 'all' && value !== ''
  ).length;

  return (
    <Card className={cn("mb-6", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-muted-foreground" />
            <h3 className="text-lg font-semibold">Filtres</h3>
            {activeFiltersCount > 0 && (
              <Badge variant="secondary" className="ml-2">
                {activeFiltersCount} actif{activeFiltersCount > 1 ? 's' : ''}
              </Badge>
            )}
          </div>
          
          {activeFiltersCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={clearFilters}
              className="gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Réinitialiser
            </Button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Search */}
          <div className="lg:col-span-2">
            <label className="text-sm font-medium mb-2 block">
              Recherche globale
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Nom, SIREN, dirigeant..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="pl-10"
              />
              {filters.search && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 p-0"
                  onClick={() => handleFilterChange('search', '')}
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>

          {/* CA Minimum */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              CA Minimum (€)
            </label>
            <div className="relative">
              <Euro className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="number"
                placeholder="0"
                value={filters.ca_min}
                onChange={(e) => handleFilterChange('ca_min', e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {/* Effectif Minimum */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Effectif Min
            </label>
            <div className="relative">
              <Users className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="number"
                placeholder="0"
                value={filters.effectif_min}
                onChange={(e) => handleFilterChange('effectif_min', e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {/* Ville */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Ville
            </label>
            <div className="relative">
              <MapPin className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground z-10" />
              <Select
                value={filters.ville}
                onChange={(e) => handleFilterChange('ville', e.target.value)}
                className="pl-10"
              >
                <option value="all">Toutes les villes</option>
                {cities.map(city => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </Select>
            </div>
          </div>

          {/* Statut */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Statut
            </label>
            <Select
              value={filters.statut}
              onChange={(e) => handleFilterChange('statut', e.target.value)}
            >
              <option value="all">Tous les statuts</option>
              <option value="prospect">Prospect</option>
              <option value="contact">Contact</option>
              <option value="qualification">Qualification</option>
              <option value="negociation">Négociation</option>
              <option value="client">Client</option>
              <option value="perdu">Perdu</option>
            </Select>
          </div>
        </div>

        {/* Active Filters Display */}
        {activeFiltersCount > 0 && (
          <div className="mt-4 pt-4 border-t">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm text-muted-foreground">Filtres actifs:</span>
              
              {filters.search && (
                <Badge variant="outline" className="gap-1">
                  <Search className="h-3 w-3" />
                  "{filters.search}"
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 ml-1"
                    onClick={() => handleFilterChange('search', '')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              )}
              
              {filters.ca_min && (
                <Badge variant="outline" className="gap-1">
                  <Euro className="h-3 w-3" />
                  CA ≥ {new Intl.NumberFormat('fr-FR').format(filters.ca_min)}€
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 ml-1"
                    onClick={() => handleFilterChange('ca_min', '')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              )}
              
              {filters.effectif_min && (
                <Badge variant="outline" className="gap-1">
                  <Users className="h-3 w-3" />
                  Effectif ≥ {filters.effectif_min}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 ml-1"
                    onClick={() => handleFilterChange('effectif_min', '')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              )}
              
              {filters.ville && filters.ville !== 'all' && (
                <Badge variant="outline" className="gap-1">
                  <MapPin className="h-3 w-3" />
                  {filters.ville}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 ml-1"
                    onClick={() => handleFilterChange('ville', 'all')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              )}
              
              {filters.statut && filters.statut !== 'all' && (
                <Badge variant="outline" className="gap-1">
                  <Filter className="h-3 w-3" />
                  {filters.statut}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-4 w-4 p-0 ml-1"
                    onClick={() => handleFilterChange('statut', 'all')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}