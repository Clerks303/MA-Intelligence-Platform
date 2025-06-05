/**
 * Modern Scraping Page - M&A Intelligence Platform
 * Professional data collection interface inspired by screenshots
 * Features: CSV import, automated scraping sources, activity timeline
 */

import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Database,
  Upload,
  FileUp,
  Settings,
  Activity,
  Download,
  Info,
  Play,
  RefreshCw,
  Zap,
  CheckCircle,
  AlertCircle,
  Clock,
  TrendingUp
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { cn } from '../lib/utils';
import api from '../services/api';

interface ScrapingSource {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  status: 'ready' | 'running' | 'error' | 'disabled';
  color: 'blue' | 'purple' | 'green' | 'orange';
  endpoint: string;
}

const scrapingSources: ScrapingSource[] = [
  {
    id: 'pappers',
    name: 'Pappers API',
    description: 'Recherche via l\'API Pappers - Données officielles',
    icon: Database,
    status: 'ready',
    color: 'blue',
    endpoint: '/scraping/pappers'
  },
  {
    id: 'societe',
    name: 'Société.com',
    description: 'Scraping du site Société.com - Données enrichies',
    icon: Activity,
    status: 'ready',
    color: 'purple',
    endpoint: '/scraping/societe'
  },
  {
    id: 'infogreffe',
    name: 'Infogreffe',
    description: 'Enrichissement Infogreffe - Données officielles',
    icon: FileUp,
    status: 'ready',
    color: 'green',
    endpoint: '/scraping/infogreffe'
  }
];

const StatusIndicator: React.FC<{ status: string }> = ({ status }) => {
  const statusConfig = {
    ready: { 
      icon: CheckCircle, 
      color: 'text-green-500',
      bg: 'bg-green-500/20',
      text: 'PRÊT'
    },
    running: { 
      icon: RefreshCw, 
      color: 'text-blue-500',
      bg: 'bg-blue-500/20',
      text: 'EN COURS'
    },
    error: { 
      icon: AlertCircle, 
      color: 'text-red-500',
      bg: 'bg-red-500/20',
      text: 'ERREUR'
    },
    disabled: { 
      icon: Clock, 
      color: 'text-slate-400',
      bg: 'bg-slate-500/20',
      text: 'INDISPONIBLE'
    }
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.ready;
  const Icon = config.icon;

  return (
    <Badge className={cn("gap-1", config.bg, config.color, "border-0")}>
      <Icon className={cn("h-3 w-3", status === 'running' && "animate-spin")} />
      {config.text}
    </Badge>
  );
};

const SourceCard: React.FC<{ 
  source: ScrapingSource; 
  onLaunch: () => void; 
  isLoading: boolean 
}> = ({ source, onLaunch, isLoading }) => {
  const Icon = source.icon;
  
  const colorClasses = {
    blue: {
      icon: 'bg-blue-500',
      button: 'border-blue-500 text-blue-400 hover:bg-blue-500/10'
    },
    purple: {
      icon: 'bg-purple-500',
      button: 'border-purple-500 text-purple-400 hover:bg-purple-500/10'
    },
    green: {
      icon: 'bg-green-500',
      button: 'border-green-500 text-green-400 hover:bg-green-500/10'
    },
    orange: {
      icon: 'bg-orange-500',
      button: 'border-orange-500 text-orange-400 hover:bg-orange-500/10'
    }
  };

  const classes = colorClasses[source.color];

  return (
    <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", classes.icon)}>
              <Icon className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 dark:text-white">
                {source.name}
              </h3>
              <StatusIndicator status={source.status} />
            </div>
          </div>
        </div>
        
        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
          {source.description}
        </p>
        
        <div className="space-y-2">
          <Button 
            variant="outline" 
            size="sm" 
            className={cn("w-full", classes.button)}
            onClick={onLaunch}
            disabled={isLoading || source.status === 'disabled'}
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                En cours...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Lancer
              </>
            )}
          </Button>
          
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full text-slate-400 hover:text-slate-900 dark:hover:text-white"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Actualiser
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

const StatsCard: React.FC<{
  title: string;
  value: string;
  subtitle: string;
  icon: React.ElementType;
  color: 'blue' | 'green' | 'purple';
}> = ({ title, value, subtitle, icon: Icon, color }) => {
  const colorClasses = {
    blue: 'bg-gradient-to-br from-blue-500 to-blue-600',
    green: 'bg-gradient-to-br from-green-500 to-green-600',
    purple: 'bg-gradient-to-br from-purple-500 to-purple-600'
  };

  return (
    <Card className={cn("border-0 text-white overflow-hidden", colorClasses[color])}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium mb-1 text-white/80">
              {title}
            </p>
            <p className="text-3xl font-bold mb-2">
              {value}
            </p>
            <p className="text-sm text-white/80">
              {subtitle}
            </p>
          </div>
          <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
            <Icon className="h-6 w-6" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const ActivityTimelineItem: React.FC<{
  title: string;
  description: string;
  time: string;
  type: 'success' | 'warning' | 'error' | 'info';
}> = ({ title, description, time, type }) => {
  const typeColors = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    info: 'bg-blue-500'
  };

  return (
    <div className="flex items-start gap-3 p-3 hover:bg-slate-50 dark:hover:bg-slate-700/50 rounded-lg transition-colors">
      <div className={cn("w-2 h-2 rounded-full mt-2 flex-shrink-0", typeColors[type])} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-900 dark:text-white">
          {title}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          {description}
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
          {time}
        </p>
      </div>
    </div>
  );
};

export const ScrapingModern: React.FC = () => {
  const queryClient = useQueryClient();
  const [dragActive, setDragActive] = useState(false);
  const [activeSource, setActiveSource] = useState<string | null>(null);

  // Upload file mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return api.post('/companies/import', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    },
    onError: (error) => {
      console.error('Upload error:', error);
    }
  });

  // Scraping mutations
  const scrapingMutations = {
    pappers: useMutation({
      mutationFn: () => api.post('/scraping/pappers'),
      onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] }),
      onSettled: () => setActiveSource(null)
    }),
    societe: useMutation({
      mutationFn: () => api.post('/scraping/societe'),
      onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] }),
      onSettled: () => setActiveSource(null)
    }),
    infogreffe: useMutation({
      mutationFn: () => api.post('/scraping/infogreffe'),
      onSuccess: () => queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] }),
      onSettled: () => setActiveSource(null)
    })
  };

  // Handle file operations
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'text/csv') {
      uploadMutation.mutate(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'text/csv') {
      uploadMutation.mutate(file);
    }
  };

  const handleSourceLaunch = (sourceId: string) => {
    setActiveSource(sourceId);
    const mutation = scrapingMutations[sourceId as keyof typeof scrapingMutations];
    mutation.mutate();
  };

  const recentActivity = [
    {
      title: 'Import CSV terminé',
      description: '23 nouvelles entreprises ajoutées depuis fichier_prospects.csv',
      time: 'Il y a 5 minutes',
      type: 'success' as const
    },
    {
      title: 'Enrichissement Pappers API',
      description: 'Enrichissement automatique de 15 entreprises via Pappers',
      time: 'Il y a 15 minutes',
      type: 'info' as const
    },
    {
      title: 'Scraping Société.com terminé',
      description: 'Collecte des données financières pour 8 entreprises',
      time: 'Il y a 1 heure',
      type: 'success' as const
    },
    {
      title: 'Limite API Pappers',
      description: 'Quota mensuel utilisé à 85% - 150 requêtes restantes',
      time: 'Il y a 2 heures',
      type: 'warning' as const
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
            <Database className="h-7 w-7 text-blue-500" />
            Collecte de données
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Importez, enrichissez et collectez automatiquement des données d'entreprises
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" className="gap-2">
            <Settings className="h-4 w-4" />
            Paramètres
          </Button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* CSV Import Section */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Upload className="h-5 w-5 text-blue-500" />
                <CardTitle>Import de données CSV</CardTitle>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Importez vos propres listes d'entreprises. Le fichier doit contenir les colonnes SIREN et nom d'entreprise.
              </p>
            </CardHeader>
            <CardContent>
              {/* Drop Zone */}
              <div
                className={cn(
                  "border-2 border-dashed rounded-lg p-12 text-center transition-colors",
                  dragActive 
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-500/10" 
                    : "border-slate-300 dark:border-slate-600 hover:border-slate-400"
                )}
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
              >
                <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                  Glissez-déposez votre fichier CSV
                </h3>
                <p className="text-slate-500 dark:text-slate-400 text-sm mb-4">
                  ou cliquez pour sélectionner un fichier
                </p>
                <p className="text-slate-400 text-xs mb-4">
                  📄 Format CSV uniquement - Max 10MB
                </p>
                
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileInput}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload">
                  <Button 
                    className="bg-blue-600 hover:bg-blue-700"
                    disabled={uploadMutation.isPending}
                  >
                    {uploadMutation.isPending ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Importation...
                      </>
                    ) : (
                      <>
                        <FileUp className="h-4 w-4 mr-2" />
                        Sélectionner le fichier
                      </>
                    )}
                  </Button>
                </label>
              </div>

              {/* Requirements */}
              <div className="mt-6 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <h4 className="font-medium text-slate-900 dark:text-white">
                    Colonnes requises
                  </h4>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <strong>Obligatoires:</strong>
                      <ul className="list-disc list-inside ml-2 mt-1">
                        <li>SIREN</li>
                        <li>Nom entreprise</li>
                      </ul>
                    </div>
                    <div>
                      <strong>Optionnelles:</strong>
                      <ul className="list-disc list-inside ml-2 mt-1">
                        <li>Email</li>
                        <li>Téléphone</li>
                        <li>Adresse</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Stats Cards */}
        <div className="space-y-4">
          <StatsCard
            title="Sources actives"
            value="3"
            subtitle="Pappers, Société, Infogreffe"
            icon={Database}
            color="blue"
          />
          
          <StatsCard
            title="Enrichissement"
            value="Auto"
            subtitle="Paramètres configurables"
            icon={TrendingUp}
            color="green"
          />
          
          <StatsCard
            title="Formats supportés"
            value="CSV"
            subtitle="Import & Export"
            icon={Download}
            color="purple"
          />
        </div>
      </div>

      {/* Information Banner */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-blue-900 dark:text-blue-300">
                  Comment utiliser la collecte de données
                </h4>
                <Badge variant="outline" className="border-blue-300 text-blue-700 dark:border-blue-600 dark:text-blue-300">
                  Guide
                </Badge>
              </div>
              <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                <p><strong>1. Importez</strong> vos listes CSV existantes avec les colonnes SIREN et nom d'entreprise</p>
                <p><strong>2. Configurez</strong> les paramètres d'enrichissement selon vos besoins</p>
                <p><strong>3. Lancez</strong> les sources de scraping pour enrichir automatiquement</p>
                <p><strong>4. Consultez</strong> la timeline pour suivre les opérations en temps réel</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scraping Sources */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              Sources de scraping automatique
            </h2>
            <p className="text-slate-500 dark:text-slate-400 mt-1">
              Collectez automatiquement des données depuis différentes sources
            </p>
          </div>
          
          <Badge variant="outline" className="gap-1">
            <span>{scrapingSources.length} sources disponibles</span>
          </Badge>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {scrapingSources.map((source) => (
            <SourceCard
              key={source.id}
              source={source}
              onLaunch={() => handleSourceLaunch(source.id)}
              isLoading={activeSource === source.id}
            />
          ))}
        </div>
      </div>

      {/* Activity Timeline */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-500" />
                Activité récente
              </CardTitle>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                Dernières opérations de collecte et d'enrichissement
              </p>
            </div>
            <Badge variant="secondary" className="gap-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              Temps réel
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recentActivity.map((item, index) => (
              <ActivityTimelineItem key={index} {...item} />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ScrapingModern;