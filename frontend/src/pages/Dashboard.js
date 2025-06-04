import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Building2, 
  TrendingUp, 
  Users, 
  Mail, 
  Phone, 
  Euro,
  RefreshCw,
  Calendar,
  BarChart3,
  Activity,
  ArrowUp,
  ArrowDown
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

import api from '../services/api';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Card } from '../components/ui/card';
import { ThemeToggle } from '../components/ui/theme-toggle';
import { cn } from '../lib/utils';

export default function Dashboard() {
  const { data: stats, isLoading, refetch, isRefetching } = useQuery(
    'stats', 
    () => api.get('/stats').then(res => res.data),
    {
      refetchInterval: 30000,
      staleTime: 25000,
    }
  );

  const formatMoney = (amount) => {
    if (!amount || amount === 0) return '0€';
    
    if (amount >= 1000000000) {
      return `${(amount / 1000000000).toFixed(1)} Mds€`;
    } else if (amount >= 1000000) {
      return `${(amount / 1000000).toFixed(1)} M€`;
    } else if (amount >= 1000) {
      return `${(amount / 1000).toFixed(0)} K€`;
    }
    return `${Math.round(amount)}€`;
  };

  const formatNumber = (num) => {
    if (!num) return '0';
    return new Intl.NumberFormat('fr-FR').format(Math.round(num));
  };

  const formatPercentage = (num) => {
    if (!num) return '0%';
    return `${num > 0 ? '+' : ''}${num.toFixed(1)}%`;
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-400" />
          <p className="text-gray-400">Chargement du tableau de bord...</p>
        </div>
      </div>
    );
  }

  // Mock status data for the pie chart since we don't have it in the response
  const statusData = [
    { name: 'A CONTACTER', value: stats?.total ? Math.floor(stats.total * 0.6) : 0, color: '#6b7280' },
    { name: 'QUALIFIE', value: stats?.total ? Math.floor(stats.total * 0.3) : 0, color: '#3b82f6' },
    { name: 'EN NEGOCIATION', value: stats?.total ? Math.floor(stats.total * 0.1) : 0, color: '#10b981' }
  ];

  // Mock trends
  const trends = {
    companies: 12,
    revenue: 8.5,
    emails: 15,
    phones: -2,
    staff: 0
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight">
              Tableau de bord
            </h1>
            <p className="text-gray-400 mt-2">
              Vue d'ensemble de votre pipeline M&A
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="gap-2 border-gray-600 text-gray-300">
              <Calendar className="h-3 w-3" />
              Mise à jour: 29/05/2025
            </Badge>
            
            <ThemeToggle />
            
            <Button 
              onClick={() => refetch()} 
              disabled={isRefetching}
              variant="outline"
              className="gap-2 border-gray-600 text-gray-300 hover:bg-gray-800"
            >
              <RefreshCw className={cn("h-4 w-4", isRefetching && "animate-spin")} />
              Actualiser
            </Button>
          </div>
        </div>

        {/* KPI Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6">
          {/* Entreprises */}
          <Card className="bg-gradient-to-br from-blue-500 to-blue-600 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium uppercase tracking-wide">ENTREPRISES</p>
                <p className="text-3xl font-bold">{formatNumber(stats?.total) || '3'}</p>
                <p className="text-blue-100 text-sm">Total dans le pipeline</p>
                <div className="flex items-center gap-1 mt-1">
                  <ArrowUp className="h-3 w-3 text-blue-200" />
                  <span className="text-blue-200 text-xs">{formatPercentage(trends.companies)}</span>
                </div>
              </div>
              <div className="bg-blue-400/20 p-3 rounded-lg">
                <Building2 className="h-8 w-8 text-blue-200" />
              </div>
            </div>
          </Card>

          {/* CA Moyen */}
          <Card className="bg-gradient-to-br from-green-500 to-green-600 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm font-medium uppercase tracking-wide">CA MOYEN</p>
                <p className="text-3xl font-bold">{formatMoney(stats?.ca_moyen) || '10.0 M€'}</p>
                <p className="text-green-100 text-sm">Par entreprise</p>
                <div className="flex items-center gap-1 mt-1">
                  <ArrowUp className="h-3 w-3 text-green-200" />
                  <span className="text-green-200 text-xs">{formatPercentage(trends.revenue)}</span>
                </div>
              </div>
              <div className="bg-green-400/20 p-3 rounded-lg">
                <Euro className="h-8 w-8 text-green-200" />
              </div>
            </div>
          </Card>

          {/* CA Total */}
          <Card className="bg-gradient-to-br from-purple-500 to-purple-600 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm font-medium uppercase tracking-wide">CA TOTAL</p>
                <p className="text-3xl font-bold">{formatMoney(stats?.ca_total) || '30.0 M€'}</p>
                <p className="text-purple-100 text-sm">Marché adressable</p>
              </div>
              <div className="bg-purple-400/20 p-3 rounded-lg">
                <TrendingUp className="h-8 w-8 text-purple-200" />
              </div>
            </div>
          </Card>

          {/* Emails */}
          <Card className="bg-gradient-to-br from-blue-600 to-blue-700 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium uppercase tracking-wide">EMAILS</p>
                <p className="text-3xl font-bold">{formatNumber(stats?.avec_email) || '3'}</p>
                <p className="text-blue-100 text-sm">100.0% de couverture</p>
                <div className="flex items-center gap-1 mt-1">
                  <ArrowUp className="h-3 w-3 text-blue-200" />
                  <span className="text-blue-200 text-xs">{formatPercentage(trends.emails)}</span>
                </div>
              </div>
              <div className="bg-blue-400/20 p-3 rounded-lg">
                <Mail className="h-8 w-8 text-blue-200" />
              </div>
            </div>
          </Card>

          {/* Téléphones */}
          <Card className="bg-gradient-to-br from-yellow-500 to-yellow-600 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-yellow-100 text-sm font-medium uppercase tracking-wide">TÉLÉPHONES</p>
                <p className="text-3xl font-bold">{formatNumber(stats?.avec_telephone) || '3'}</p>
                <p className="text-yellow-100 text-sm">100.0% de couverture</p>
                <div className="flex items-center gap-1 mt-1">
                  <ArrowDown className="h-3 w-3 text-yellow-200" />
                  <span className="text-yellow-200 text-xs">{formatPercentage(trends.phones)}</span>
                </div>
              </div>
              <div className="bg-yellow-400/20 p-3 rounded-lg">
                <Phone className="h-8 w-8 text-yellow-200" />
              </div>
            </div>
          </Card>

          {/* Effectif Moyen */}
          <Card className="bg-gradient-to-br from-red-500 to-red-600 border-0 p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-100 text-sm font-medium uppercase tracking-wide">EFFECTIF MOYEN</p>
                <p className="text-3xl font-bold">{formatNumber(stats?.effectif_moyen) || '0'}</p>
                <p className="text-red-100 text-sm">Collaborateurs</p>
              </div>
              <div className="bg-red-400/20 p-3 rounded-lg">
                <Users className="h-8 w-8 text-red-200" />
              </div>
            </div>
          </Card>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pie Chart */}
          <Card className="bg-gray-800 border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-semibold text-white">Répartition par statut</h3>
                <p className="text-gray-400 text-sm">Distribution des entreprises selon leur stade</p>
              </div>
              <Badge variant="secondary" className="gap-1 bg-gray-700 text-gray-300 border-gray-600">
                <BarChart3 className="h-3 w-3" />
                1 statuts
              </Badge>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    animationDuration={800}
                  >
                    {statusData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.color}
                        className="hover:opacity-80 transition-opacity cursor-pointer"
                      />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#ffffff'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Bar Chart */}
          <Card className="bg-gray-800 border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-semibold text-white">Pipeline de prospection</h3>
                <p className="text-gray-400 text-sm">Volume d'entreprises par statut</p>
              </div>
              <Badge variant="secondary" className="gap-1 bg-gray-700 text-gray-300 border-gray-600">
                <BarChart3 className="h-3 w-3" />
                Analyse détaillée
              </Badge>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[{ name: 'EN PROSPECT', value: 3 }]} margin={{ bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45} 
                    textAnchor="end" 
                    height={80}
                    fontSize={12}
                    stroke="#9ca3af"
                  />
                  <YAxis stroke="#9ca3af" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#ffffff'
                    }}
                  />
                  <Bar 
                    dataKey="value" 
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                    className="hover:opacity-80 transition-opacity"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Activity Section */}
        <Card className="bg-gray-800 border-gray-700 p-6">
          <div className="flex items-center gap-2 mb-6">
            <Activity className="h-6 w-6 text-blue-400" />
            <h2 className="text-2xl font-semibold text-white">Activité récente</h2>
          </div>
          
          <div className="text-center py-12">
            <Activity className="h-12 w-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">Aucune activité récente</p>
            <p className="text-gray-500 text-sm mt-1">
              Les dernières actions et événements apparaîtront ici
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}