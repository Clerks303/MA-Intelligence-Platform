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
  MoreVertical,
  Star,
  Calendar
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

  // Queries
  // Transform filters for API call
  const transformFilters = (filters) => {
    const transformed = {};
    
    // Only add non-empty values, convert types properly
    if (filters.ca_min && filters.ca_min !== '') {
      transformed.ca_min = parseFloat(filters.ca_min);
    }
    if (filters.effectif_min && filters.effectif_min !== '') {
      transformed.effectif_min = parseInt(filters.effectif_min, 10);
    }
    if (filters.ville && filters.ville !== 'all') {
      transformed.ville = filters.ville;
    }
    if (filters.statut && filters.statut !== 'all') {
      transformed.statut = filters.statut;
    }
    if (filters.search && filters.search !== '') {
      transformed.search = filters.search;
    }
    
    return transformed;
  };

  const { data: companies = [], isLoading } = useQuery(
    ['companies', filters],
    () => api.post('/companies/filter', transformFilters(filters)).then(res => res.data),
    {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: false,
    }
  );

  const { data: cities = [] } = useQuery(
    'cities',
    () => api.get('/stats/cities').then(res => res.data.cities),
    {
      staleTime: 300000, // 5 minutes
    }
  );

  // Mutations
  const deleteMutation = useMutation(
    (siren) => api.delete(`/companies/${siren}`),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('companies');
        queryClient.invalidateQueries('stats');
      }
    }
  );

  const createMutation = useMutation(
    (companyData) => api.post('/companies/', companyData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('companies');
        queryClient.invalidateQueries('stats');
      }
    }
  );

  const updateMutation = useMutation(
    ({ siren, data }) => api.put(`/companies/${siren}`, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('companies');
        queryClient.invalidateQueries('stats');
      }
    }
  );

  // Format functions
  const formatCurrency = (amount) => {
    if (!amount || amount === 0) return '-';
    return new Intl.NumberFormat('fr-FR', { 
      style: 'currency', 
      currency: 'EUR',
      notation: amount >= 1000000 ? 'compact' : 'standard',
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return format(new Date(dateString), 'dd/MM/yyyy', { locale: fr });
    } catch {
      return '-';
    }
  };

  // Table columns
  const columns = [
    {
      accessorKey: 'statut',
      header: 'Statut',
      width: '120px',
      cell: ({ getValue }) => <StatusBadge status={getValue()} />,
      sortable: true,
    },
    {
      accessorKey: 'score_prospection',
      header: 'Score',
      width: '100px',
      cell: ({ getValue }) => {
        const score = getValue();
        if (!score) return '-';
        return (
          <div className="flex items-center gap-1">
            <Star className="h-3 w-3 text-yellow-500" />
            <span className="font-medium">{Math.round(score)}%</span>
          </div>
        );
      },
      sortable: true,
    },
    {
      accessorKey: 'nom_entreprise',
      header: 'Entreprise',
      cell: ({ getValue, row }) => (
        <div className="font-medium max-w-[250px] truncate" title={getValue()}>
          {getValue()}
        </div>
      ),
      sortable: true,
    },
    {
      accessorKey: 'siren',
      header: 'SIREN',
      width: '110px',
      cell: ({ getValue }) => (
        <span className="font-mono text-sm">{getValue() || "N/A"}</span>
      ),
      sortable: true,
    },
    {
      accessorKey: 'chiffre_affaires',
      header: 'CA',
      width: '120px',
      cell: ({ getValue }) => (
        <div className="flex items-center gap-1">
          <Euro className="h-3 w-3 text-muted-foreground" />
          <span className="font-medium">{formatCurrency(getValue())}</span>
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
        if (!effectif) return '-';
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
      cell: ({ getValue }) => (
        <span className="max-w-[180px] truncate block" title={getValue()}>
          {getValue() || '-'}
        </span>
      ),
      sortable: true,
    },
    {
      accessorKey: 'email',
      header: 'Contact',
      width: '180px',
      cell: ({ getValue, row }) => {
        const email = getValue();
        const phone = row.original?.telephone;
        
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
          {formatDate(getValue())}
        </div>
      ),
      sortable: true,
    },
    {
      accessorKey: 'actions',
      header: 'Actions',
      width: '120px',
      sortable: false,
      cell: ({ row }) => (
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setSelectedCompany(row.original?.siren);
              setDetailsOpen(true);
            }}
            className="h-8 w-8 p-0"
          >
            <Eye className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setCompanyToEdit(row.original);
              setEditDialogOpen(true);
            }}
            className="h-8 w-8 p-0"
          >
            <Edit className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              if (window.confirm('Êtes-vous sûr de vouloir supprimer cette entreprise ?')) {
                deleteMutation.mutate(row.original?.siren);
              }
            }}
            className="h-8 w-8 p-0 text-destructive hover:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      ),
    },
  ];

  // Export function
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
    }
  };

  const handleRowClick = (row) => {
    setSelectedCompany(row?.siren);
    setDetailsOpen(true);
  };

  const activeFiltersCount = Object.values(filters).filter(value => 
    value && value !== 'all' && value !== ''
  ).length;

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
            {companies.length > 0 && (
              <div className="flex items-center gap-4 mt-3">
                <Badge variant="secondary" className="gap-1">
                  <Building2 className="h-3 w-3" />
                  {companies.length} entreprise{companies.length > 1 ? 's' : ''}
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
              disabled={companies.length === 0}
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
            data={companies}
            columns={columns}
            searchable={false} // Search is handled by filters
            pageSize={25}
            loading={isLoading}
            onRowClick={handleRowClick}
            className="border rounded-lg"
          />
          
          {!isLoading && companies.length === 0 && (
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
        {companies.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-950 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Total entreprises</p>
                  <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                    {companies.length}
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
                    {companies.filter(c => c?.email).length}
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
                    {companies.filter(c => c?.telephone).length}
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
                    {Math.round(
                      companies.reduce((acc, c) => acc + (c?.score_prospection || 0), 0) / 
                      companies.filter(c => c?.score_prospection).length
                    ) || 0}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-orange-500" />
              </div>
            </div>
          </div>
        )}

        {/* Company Details Dialog */}
        <CompanyDetailsDialog
          open={detailsOpen}
          onClose={() => {
            setDetailsOpen(false);
            setSelectedCompany(null);
          }}
          siren={selectedCompany}
        />

        {/* Add Company Dialog */}
        <AddCompanyDialog
          open={addDialogOpen}
          onClose={() => setAddDialogOpen(false)}
        />

        {/* Edit Company Dialog */}
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