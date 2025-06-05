import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Building2,
  Download,
  Eye,
  Edit,
  Trash2,
  Euro,
  Users,
  Mail,
  Phone,
  TrendingUp,
  Plus,
  Star,
  Calendar,
  AlertTriangle,
  RefreshCw
} from 'lucide-react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

import { DataTable } from '../components/ui/data-table';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { ThemeToggle } from '../components/ui/theme-toggle';
import { CompanyFilters } from '../components/companies/CompanyFilters';
import { StatusBadge } from '../components/companies/StatusBadge';
import { CompanyDetailsDialog } from '../components/companies/CompanyDetailsDialog';
import { AddCompanyDialog } from '../components/companies/AddCompanyDialog';
import { EditCompanyDialog } from '../components/companies/EditCompanyDialog';
import api from '../services/api';
import { cn } from '../lib/utils';

export default function Companies() {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState({
    ca_min: '',
    effectif_min: '',
    ville: 'all',
    statut: 'all',
    search: ''
  });
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [companyToEdit, setCompanyToEdit] = useState(null);

  // 🔒 SECURISATION : Transform filters avec vérifications strictes
  const transformFilters = (filters) => {
    const transformed = {};
    
    try {
      // Vérifications strictes pour éviter les erreurs de type
      if (filters?.ca_min && filters.ca_min !== '' && !isNaN(filters.ca_min)) {
        transformed.ca_min = parseFloat(filters.ca_min);
      }
      if (filters?.effectif_min && filters.effectif_min !== '' && !isNaN(filters.effectif_min)) {
        transformed.effectif_min = parseInt(filters.effectif_min, 10);
      }
      if (filters?.ville && filters.ville !== 'all' && filters.ville !== '') {
        transformed.ville = String(filters.ville);
      }
      if (filters?.statut && filters.statut !== 'all' && filters.statut !== '') {
        transformed.statut = String(filters.statut);
      }
      if (filters?.search && filters.search !== '') {
        transformed.search = String(filters.search);
      }
    } catch (error) {
      console.error('Erreur lors de la transformation des filtres:', error);
    }
    
    return transformed;
  };

  // 🔒 SECURISATION : Queries avec gestion d'erreur robuste
  const { data: companies = [], isLoading, error: companiesError, refetch } = useQuery({
    queryKey: ['companies', filters],
    queryFn: () => api.post('/companies/filter', transformFilters(filters)).then(res => res.data),
    staleTime: 30000,
    refetchOnWindowFocus: false,
    retry: 2,
    retryDelay: 1000,
    onError: (error) => {
      console.error('Erreur lors du chargement des entreprises:', error);
    }
  });

  const { data: cities = [] } = useQuery({
    queryKey: ['cities'],
    queryFn: () => api.get('/stats/cities').then(res => res.data?.cities || []),
    staleTime: 300000,
    retry: 2,
    onError: (error) => {
      console.error('Erreur lors du chargement des villes:', error);
    }
  });

  // 🔒 SECURISATION : Mutations avec gestion d'erreur
  const deleteMutation = useMutation({
    mutationFn: (siren) => {
      if (!siren) throw new Error('SIREN manquant');
      return api.delete(`/companies/${siren}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: (error) => {
      console.error('Erreur lors de la suppression:', error);
      alert('Erreur lors de la suppression de l\'entreprise');
    }
  });

  const createMutation = useMutation({
    mutationFn: (companyData) => {
      if (!companyData) throw new Error('Données manquantes');
      return api.post('/companies/', companyData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: (error) => {
      console.error('Erreur lors de la création:', error);
    }
  });

  const updateMutation = useMutation({
    mutationFn: ({ siren, data }) => {
      if (!siren || !data) throw new Error('Données manquantes');
      return api.put(`/companies/${siren}`, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: (error) => {
      console.error('Erreur lors de la mise à jour:', error);
    }
  });

  // 🔒 SECURISATION : Fonctions de formatage sécurisées
  const safeFormatCurrency = (amount) => {
    if (amount === null || amount === undefined || amount === '' || isNaN(amount)) {
      return '-';
    }
    try {
      const numAmount = Number(amount);
      return new Intl.NumberFormat('fr-FR', { 
        style: 'currency', 
        currency: 'EUR',
        notation: numAmount >= 1000000 ? 'compact' : 'standard',
        maximumFractionDigits: 0
      }).format(numAmount);
    } catch (error) {
      console.error('Erreur formatage currency:', error);
      return '-';
    }
  };

  const safeFormatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return '-';
      return format(date, 'dd/MM/yyyy', { locale: fr });
    } catch (error) {
      console.error('Erreur formatage date:', error);
      return '-';
    }
  };

  const safeGetValue = (obj, key, defaultValue = '-') => {
    try {
      return obj?.[key] ?? defaultValue;
    } catch (error) {
      console.error(`Erreur accès propriété ${key}:`, error);
      return defaultValue;
    }
  };

  // 🔒 SECURISATION : Gestion d'erreur pour l'état global
  if (companiesError) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4 max-w-md text-center">
          <AlertTriangle className="h-12 w-12 text-red-500" />
          <h2 className="text-xl font-semibold">Erreur de chargement</h2>
          <p className="text-muted-foreground">
            Impossible de charger la liste des entreprises.
          </p>
          <div className="flex gap-2">
            <Button
              onClick={() => refetch()}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Réessayer
            </Button>
            <Button
              variant="outline"
              onClick={() => setFilters({
                ca_min: '',
                effectif_min: '',
                ville: 'all',
                statut: 'all',
                search: ''
              })}
            >
              Réinitialiser filtres
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // 🔒 SECURISATION : Colonnes du tableau avec vérifications strictes
  const columns = [
    {
      accessorKey: 'statut',
      header: 'Statut',
      width: '120px',
      cell: ({ getValue }) => {
        const status = getValue();
        return status ? <StatusBadge status={status} /> : <span className="text-muted-foreground">-</span>;
      },
      sortable: true,
    },
    {
      accessorKey: 'score_prospection',
      header: 'Score',
      width: '100px',
      cell: ({ getValue }) => {
        const score = getValue();
        if (!score || isNaN(score)) return <span className="text-muted-foreground">-</span>;
        return (
          <div className="flex items-center gap-1">
            <Star className="h-3 w-3 text-yellow-500" />
            <span className="font-medium">{Math.round(Number(score))}%</span>
          </div>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'nom_entreprise',
      header: 'Entreprise',
      cell: ({ getValue, row }) => {
        const name = getValue();
        return (
          <div className="font-medium max-w-[250px] truncate" title={name || 'N/A'}>
            {name || 'N/A'}
          </div>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'siren',
      header: 'SIREN',
      width: '110px',
      cell: ({ getValue }) => {
        const siren = getValue();
        return (
          <span className="font-mono text-sm">{siren || "N/A"}</span>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'chiffre_affaires',
      header: 'CA',
      width: '120px',
      cell: ({ getValue }) => (
        <div className="flex items-center gap-1">
          <Euro className="h-3 w-3 text-muted-foreground" />
          <span className="font-medium">{safeFormatCurrency(getValue())}</span>
        </div>
      ),
      sortable: true,
    },
    {
      accessorKey: 'effectif',
      header: 'Effectif',
      width: '100px',
      cell: ({ getValue }) => {
        const effectif = getValue();
        if (!effectif || isNaN(effectif)) return <span className="text-muted-foreground">-</span>;
        return (
          <div className="flex items-center gap-1">
            <Users className="h-3 w-3 text-muted-foreground" />
            <span>{effectif}</span>
          </div>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'dirigeant_principal',
      header: 'Dirigeant',
      width: '200px',
      cell: ({ getValue }) => {
        const dirigeant = getValue();
        return (
          <span className="max-w-[180px] truncate block" title={dirigeant || '-'}>
            {dirigeant || '-'}
          </span>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'email',
      header: 'Contact',
      width: '180px',
      cell: ({ getValue, row }) => {
        const email = getValue();
        const phone = safeGetValue(row.original, 'telephone', null);
        
        return (
          <div className="space-y-1">
            {email && (
              <div className="flex items-center gap-1 text-xs">
                <Mail className="h-3 w-3 text-muted-foreground" />
                <a 
                  href={`mailto:${email}`} 
                  className="text-primary hover:underline truncate max-w-[140px]"
                  title={email}
                >
                  {email}
                </a>
              </div>
            )}
            {phone && (
              <div className="flex items-center gap-1 text-xs">
                <Phone className="h-3 w-3 text-muted-foreground" />
                <a 
                  href={`tel:${phone}`} 
                  className="text-primary hover:underline"
                >
                  {phone}
                </a>
              </div>
            )}
            {!email && !phone && <span className="text-muted-foreground">-</span>}
          </div>
        );
      },
      sortable: false,
    },
    {
      accessorKey: 'date_creation',
      header: 'Création',
      width: '110px',
      cell: ({ getValue }) => (
        <div className="flex items-center gap-1 text-sm">
          <Calendar className="h-3 w-3 text-muted-foreground" />
          {safeFormatDate(getValue())}
        </div>
      ),
      sortable: true,
    },
    {
      accessorKey: 'actions',
      header: 'Actions',
      width: '120px',
      sortable: false,
      cell: ({ row }) => {
        const rowData = row?.original;
        if (!rowData) return null;

        return (
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                const siren = safeGetValue(rowData, 'siren');
                if (siren && siren !== '-') {
                  setSelectedCompany(siren);
                  setDetailsOpen(true);
                }
              }}
              className="h-8 w-8 p-0"
              title="Voir les détails"
            >
              <Eye className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                setCompanyToEdit(rowData);
                setEditDialogOpen(true);
              }}
              className="h-8 w-8 p-0"
              title="Modifier"
            >
              <Edit className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                const siren = safeGetValue(rowData, 'siren');
                const name = safeGetValue(rowData, 'nom_entreprise', 'cette entreprise');
                if (siren && siren !== '-' && window.confirm(`Êtes-vous sûr de vouloir supprimer ${name} ?`)) {
                  deleteMutation.mutate(siren);
                }
              }}
              className="h-8 w-8 p-0 text-destructive hover:text-destructive"
              title="Supprimer"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        );
      },
    },
  ];

  // 🔒 SECURISATION : Export avec gestion d'erreur
  const handleExport = async () => {
    try {
      const response = await api.post('/companies/export', transformFilters(filters), {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `entreprises_${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Erreur lors de l\'export:', error);
      alert('Erreur lors de l\'export des données');
    }
  };

  const handleRowClick = (row) => {
    if (!row) return;
    const siren = safeGetValue(row, 'siren');
    if (siren && siren !== '-') {
      setSelectedCompany(siren);
      setDetailsOpen(true);
    }
  };

  // 🔒 SECURISATION : Calcul sécurisé des filtres actifs
  const activeFiltersCount = Object.values(filters || {}).filter(value => 
    value && value !== 'all' && value !== ''
  ).length;

  // 🔒 SECURISATION : Stats sécurisées
  const safeCompanies = Array.isArray(companies) ? companies : [];
  const safeStats = {
    total: safeCompanies.length,
    withEmail: safeCompanies.filter(c => c?.email).length,
    withPhone: safeCompanies.filter(c => c?.telephone).length,
    avgScore: safeCompanies.length > 0 
      ? Math.round(
          safeCompanies.reduce((acc, c) => acc + (Number(c?.score_prospection) || 0), 0) / 
          safeCompanies.filter(c => c?.score_prospection && !isNaN(c.score_prospection)).length
        ) || 0
      : 0
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-8 space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3">
              <Building2 className="h-8 w-8 text-primary" />
              Bibliothèque d'entreprises
            </h1>
            <p className="text-muted-foreground mt-2">
              Gérez et explorez votre base de données d'entreprises
            </p>
            {safeStats.total > 0 && (
              <div className="flex items-center gap-4 mt-3">
                <Badge variant="secondary" className="gap-1">
                  <Building2 className="h-3 w-3" />
                  {safeStats.total} entreprise{safeStats.total > 1 ? 's' : ''}
                </Badge>
                {activeFiltersCount > 0 && (
                  <Badge variant="outline">
                    {activeFiltersCount} filtre{activeFiltersCount > 1 ? 's' : ''} actif{activeFiltersCount > 1 ? 's' : ''}
                  </Badge>
                )}
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-3">
            <ThemeToggle />
            
            <Button
              variant="outline"
              onClick={handleExport}
              disabled={safeStats.total === 0}
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              Exporter CSV
            </Button>
            
            <Button className="gap-2" onClick={() => setAddDialogOpen(true)}>
              <Plus className="h-4 w-4" />
              Ajouter
            </Button>
          </div>
        </div>

        {/* Filters */}
        <CompanyFilters
          filters={filters}
          onFiltersChange={setFilters}
          cities={cities}
        />

        {/* Data Table */}
        <div className="space-y-4">
          <DataTable
            data={safeCompanies}
            columns={columns}
            searchable={false} // Search is handled by filters
            pageSize={25}
            loading={isLoading}
            onRowClick={handleRowClick}
            className="border rounded-lg"
          />
          
          {!isLoading && safeStats.total === 0 && (
            <div className="text-center py-12 border rounded-lg bg-muted/20">
              <Building2 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Aucune entreprise trouvée</h3>
              <p className="text-muted-foreground mb-4">
                {activeFiltersCount > 0 
                  ? 'Aucune entreprise ne correspond à vos critères de recherche.'
                  : 'Commencez par importer des entreprises ou ajustez vos filtres.'
                }
              </p>
              {activeFiltersCount > 0 && (
                <Button 
                  variant="outline" 
                  onClick={() => setFilters({
                    ca_min: '',
                    effectif_min: '',
                    ville: 'all',
                    statut: 'all',
                    search: ''
                  })}
                >
                  Réinitialiser les filtres
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Quick Stats */}
        {safeStats.total > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-950 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Total entreprises</p>
                  <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                    {safeStats.total}
                  </p>
                </div>
                <Building2 className="h-8 w-8 text-blue-500" />
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-950 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600 dark:text-green-400">Avec email</p>
                  <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                    {safeStats.withEmail}
                  </p>
                </div>
                <Mail className="h-8 w-8 text-green-500" />
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-950 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600 dark:text-purple-400">Avec téléphone</p>
                  <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                    {safeStats.withPhone}
                  </p>
                </div>
                <Phone className="h-8 w-8 text-purple-500" />
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-950 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-600 dark:text-orange-400">Score moyen</p>
                  <p className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                    {safeStats.avgScore}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-orange-500" />
              </div>
            </div>
          </div>
        )}

        {/* 🔒 SECURISATION : Dialogs avec vérifications */}
        <CompanyDetailsDialog
          open={detailsOpen}
          onClose={() => {
            setDetailsOpen(false);
            setSelectedCompany(null);
          }}
          siren={selectedCompany}
        />

        <AddCompanyDialog
          open={addDialogOpen}
          onClose={() => setAddDialogOpen(false)}
        />

        <EditCompanyDialog
          open={editDialogOpen}
          onClose={() => {
            setEditDialogOpen(false);
            setCompanyToEdit(null);
          }}
          company={companyToEdit}
        />
      </div>
    </div>
  );
}