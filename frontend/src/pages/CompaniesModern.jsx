/**
 * Modern Companies Page - M&A Intelligence Platform
 * Professional company management with filters, table, and actions
 * Inspired by the UI patterns shown in screenshots
 */

import React, { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Building2, 
  Plus, 
  Search, 
  Filter,
  Download,
  Upload,
  Mail,
  Phone,
  MapPin,
  Euro,
  Users,
  Calendar,
  ExternalLink,
  MoreHorizontal,
  Edit,
  Trash2,
  Eye,
  RefreshCw
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { DataTable } from '../components/ui/data-table';
import { cn } from '../lib/utils';
import api from '../services/api';

const StatusBadge = ({ status }) => {
  const statusConfig = {
    'à contacter': { variant: 'default', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400' },
    'contacté': { variant: 'secondary', color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' },
    'qualifié': { variant: 'success', color: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' },
    'non qualifié': { variant: 'destructive', color: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400' },
    'STATUT_PROSPECTION': { variant: 'default', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400' }
  };

  const config = statusConfig[status] || statusConfig['à contacter'];

  return (
    <Badge className={cn("text-xs font-medium", config.color)}>
      {status}
    </Badge>
  );
};

const CompanyActionsMenu = ({ company, onEdit, onDelete }) => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setMenuOpen(!menuOpen)}
        className="h-8 w-8 p-0"
      >
        <MoreHorizontal className="h-4 w-4" />
      </Button>
      
      {menuOpen && (
        <div className="absolute right-0 top-full mt-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-50 min-w-[160px]">
          <div className="py-1">
            <button
              onClick={() => { onEdit(); setMenuOpen(false); }}
              className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
            >
              <Edit className="h-4 w-4" />
              Modifier
            </button>
            <button
              onClick={() => { onDelete(); setMenuOpen(false); }}
              className="w-full text-left px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-2"
            >
              <Trash2 className="h-4 w-4" />
              Supprimer
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export const CompaniesModern = () => {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState({
    search: '',
    statut: '',
    minCa: '',
    maxCa: '',
    ville: ''
  });
  const [selectedCompanies, setSelectedCompanies] = useState([]);

  // Fetch companies
  const { data: companies = [], isLoading, refetch } = useQuery({
    queryKey: ['companies'],
    queryFn: () => api.get('/companies').then(res => res.data),
    refetchInterval: 30000,
  });

  // Delete company mutation
  const deleteMutation = useMutation({
    mutationFn: (id) => api.delete(`/companies/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
    },
  });

  // Apply filters to companies
  const filteredCompanies = useMemo(() => {
    return companies.filter((company) => {
      const matchesSearch = !filters.search || 
        company.nom_entreprise?.toLowerCase().includes(filters.search.toLowerCase()) ||
        company.siren?.includes(filters.search) ||
        company.email?.toLowerCase().includes(filters.search.toLowerCase());
      
      const matchesStatus = !filters.statut || company.statut === filters.statut;
      
      const matchesVille = !filters.ville || 
        company.ville?.toLowerCase().includes(filters.ville.toLowerCase());
      
      const matchesMinCa = !filters.minCa || 
        !company.chiffre_affaires || company.chiffre_affaires >= parseFloat(filters.minCa);
      
      const matchesMaxCa = !filters.maxCa || 
        !company.chiffre_affaires || company.chiffre_affaires <= parseFloat(filters.maxCa);

      return matchesSearch && matchesStatus && matchesVille && matchesMinCa && matchesMaxCa;
    });
  }, [companies, filters]);

  // Table columns configuration
  const columns = [
    {
      accessorKey: 'nom_entreprise',
      header: 'Entreprise',
      cell: ({ row }) => (
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
            <Building2 className="h-4 w-4 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <div className="font-medium text-slate-900 dark:text-white">
              {row.original.nom_entreprise}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              SIREN: {row.original.siren}
            </div>
          </div>
        </div>
      ),
    },
    {
      accessorKey: 'statut',
      header: 'Statut',
      cell: ({ getValue }) => (
        <StatusBadge status={getValue()} />
      ),
    },
    {
      accessorKey: 'chiffre_affaires',
      header: 'CA',
      cell: ({ getValue }) => {
        const ca = getValue();
        return ca ? (
          <div className="flex items-center gap-1 text-sm">
            <Euro className="h-3 w-3 text-green-600" />
            <span className="font-medium">{(ca / 1000000).toFixed(1)}M€</span>
          </div>
        ) : (
          <span className="text-slate-400 text-sm">-</span>
        );
      },
    },
    {
      accessorKey: 'effectif',
      header: 'Effectif',
      cell: ({ getValue }) => {
        const effectif = getValue();
        return effectif ? (
          <div className="flex items-center gap-1 text-sm">
            <Users className="h-3 w-3 text-blue-600" />
            <span>{effectif}</span>
          </div>
        ) : (
          <span className="text-slate-400 text-sm">-</span>
        );
      },
    },
    {
      accessorKey: 'ville',
      header: 'Localisation',
      cell: ({ row }) => (
        <div className="text-sm">
          {row.original.ville && (
            <div className="flex items-center gap-1">
              <MapPin className="h-3 w-3 text-slate-400" />
              <span>{row.original.ville}</span>
              {row.original.code_postal && (
                <span className="text-slate-500">({row.original.code_postal})</span>
              )}
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: 'email',
      header: 'Contact',
      cell: ({ row }) => (
        <div className="flex items-center gap-2">
          {row.original.email && (
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
              <Mail className="h-3 w-3 text-blue-600" />
            </Button>
          )}
          {row.original.telephone && (
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
              <Phone className="h-3 w-3 text-green-600" />
            </Button>
          )}
        </div>
      ),
    },
    {
      id: 'actions',
      header: '',
      cell: ({ row }) => (
        <CompanyActionsMenu
          company={row.original}
          onEdit={() => console.log('Edit', row.original.id)}
          onDelete={() => console.log('Delete', row.original.id)}
        />
      ),
    },
  ];

  const stats = useMemo(() => ({
    total: companies.length,
    filtered: filteredCompanies.length,
    contacted: companies.filter((c: Company) => c.statut === 'contacté').length,
    qualified: companies.filter((c: Company) => c.statut === 'qualifié').length,
  }), [companies, filteredCompanies]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Entreprises
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Gestion des entreprises prospects - {stats.filtered} entreprises affichées
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="gap-2">
            <Upload className="h-4 w-4" />
            Importer CSV
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Exporter
          </Button>
          <Button className="gap-2 bg-blue-600 hover:bg-blue-700">
            <Plus className="h-4 w-4" />
            Ajouter entreprise
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-lg flex items-center justify-center">
                <Building2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {stats.total}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Total entreprises
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-yellow-100 dark:bg-yellow-900/20 rounded-lg flex items-center justify-center">
                <Mail className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {stats.contacted}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Contactées
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
                <Users className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {stats.qualified}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Qualifiées
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/20 rounded-lg flex items-center justify-center">
                <Euro className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {companies.reduce((sum: number, c: Company) => sum + (c.chiffre_affaires || 0), 0) / 1000000}M€
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  CA total pipeline
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-slate-600 dark:text-slate-400" />
            <CardTitle className="text-lg">Filtres</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                Recherche
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <Input
                  placeholder="Nom, SIREN, email..."
                  value={filters.search}
                  onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                Statut
              </label>
              <select
                value={filters.statut}
                onChange={(e) => setFilters(prev => ({ ...prev, statut: e.target.value }))}
                className="input w-full"
              >
                <option value="">Tous les statuts</option>
                <option value="à contacter">À contacter</option>
                <option value="contacté">Contacté</option>
                <option value="qualifié">Qualifié</option>
                <option value="non qualifié">Non qualifié</option>
                <option value="STATUT_PROSPECTION">Prospection</option>
              </select>
            </div>

            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                Ville
              </label>
              <Input
                placeholder="Paris, Lyon..."
                value={filters.ville}
                onChange={(e) => setFilters(prev => ({ ...prev, ville: e.target.value }))}
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                CA min (M€)
              </label>
              <Input
                type="number"
                placeholder="1.0"
                value={filters.minCa}
                onChange={(e) => setFilters(prev => ({ ...prev, minCa: e.target.value }))}
              />
            </div>

            <div>
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                CA max (M€)
              </label>
              <Input
                type="number"
                placeholder="100.0"
                value={filters.maxCa}
                onChange={(e) => setFilters(prev => ({ ...prev, maxCa: e.target.value }))}
              />
            </div>
          </div>

          <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
            <div className="text-sm text-slate-500 dark:text-slate-400">
              {stats.filtered} entreprises correspondent aux filtres
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFilters({ search: '', statut: '', minCa: '', maxCa: '', ville: '' })}
              className="text-slate-600 dark:text-slate-400"
            >
              Réinitialiser les filtres
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Companies Table */}
      <Card>
        <CardContent className="p-0">
          <DataTable
            data={filteredCompanies}
            columns={columns}
            searchable={false}
            pageSize={20}
            loading={isLoading}
            onRowClick={(company) => console.log('View company:', company)}
            className=""
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default CompaniesModern;