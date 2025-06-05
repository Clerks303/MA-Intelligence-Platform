import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Database,
  Upload,
  FileUp,
  CheckCircle,
  Settings,
  Activity,
  TrendingUp,
  Download,
  Info,
  Play,
  RefreshCcw,
  Zap
} from 'lucide-react';

import { ThemeToggle } from '../components/ui/theme-toggle';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Card } from '../components/ui/card';
import api from '../services/api';

export default function Scraping() {
  const queryClient = useQueryClient();
  const [dragActive, setDragActive] = useState(false);
  const [enrichmentParams, setEnrichmentParams] = useState({
    min_ca: 1000000,
    min_score: 50,
    siren: ''
  });

  // Upload file mutation
  const uploadMutation = useMutation({
    mutationFn: async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      return api.post('/companies/import', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: (error) => {
      console.error('Upload error:', error);
    }
  });

  // Start scraping mutations
  const pappersMutation = useMutation({
    mutationFn: () => api.post('/scraping/pappers'),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['stats'] })
  });

  const societeMutation = useMutation({
    mutationFn: () => api.post('/scraping/societe'),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['stats'] })
  });

  const infogreffeMutation = useMutation({
    mutationFn: () => api.post('/scraping/infogreffe'),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['stats'] })
  });

  // Handle file drop
  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'text/csv') {
      uploadMutation.mutate(files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  // Handle file input change
  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'text/csv') {
      uploadMutation.mutate(file);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight flex items-center gap-3">
              <Database className="h-8 w-8 text-blue-400" />
              Collecte de données
            </h1>
            <p className="text-gray-400 mt-2">
              Importez, enrichissez et collectez automatiquement des données d'entreprises
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <Button
              variant="outline"
              className="gap-2 border-gray-600 text-gray-300 hover:bg-gray-800"
            >
              <Settings className="h-4 w-4" />
              Reset paramètres
            </Button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* CSV Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <Upload className="h-5 w-5 text-blue-400" />
                <h2 className="text-xl font-semibold">Import de données CSV</h2>
              </div>
              <p className="text-gray-400 text-sm mb-6">
                Importez vos propres listes d'entreprises. Le fichier doit contenir les colonnes SIREN et nom d'entreprise.
              </p>

              {/* Drop Zone */}
              <div
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  dragActive 
                    ? 'border-blue-500 bg-blue-500/10' 
                    : 'border-gray-600 hover:border-gray-500'
                }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">
                  Glissez-déposez votre fichier CSV
                </h3>
                <p className="text-gray-400 text-sm mb-4">
                  ou cliquez pour sélectionner un fichier
                </p>
                <p className="text-gray-500 text-xs mb-4">
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
                    variant="default" 
                    className="bg-blue-600 hover:bg-blue-700"
                    disabled={uploadMutation.isLoading}
                  >
                    {uploadMutation.isLoading ? (
                      <>
                        <RefreshCcw className="h-4 w-4 mr-2 animate-spin" />
                        Importation...
                      </>
                    ) : (
                      <>
                        <FileUp className="h-4 w-4 mr-2" />
                        Importer le fichier
                      </>
                    )}
                  </Button>
                </label>
              </div>

              {/* Update existing companies section */}
              <div className="mt-6 p-4 bg-gray-700/50 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="h-4 w-4 text-green-400" />
                  <h4 className="font-medium">Mettre à jour les entreprises existantes</h4>
                </div>
                <div className="text-sm text-gray-400 space-y-1">
                  <p><strong>Colonnes requises :</strong></p>
                  <ul className="list-disc list-inside ml-4 space-y-1">
                    <li>SIREN (obligatoire)</li>
                    <li>Nom entreprise (obligatoire)</li>
                    <li>Email, téléphone, adresse (optionnel)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Stats Cards */}
          <div className="space-y-4">
            {/* Sources actives */}
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm font-medium">Sources actives</p>
                  <p className="text-3xl font-bold">3</p>
                  <p className="text-blue-100 text-sm">Pappers, Société, Infogreffe</p>
                </div>
                <Database className="h-8 w-8 text-blue-200" />
              </div>
            </div>

            {/* Enrichissement */}
            <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100 text-sm font-medium">Enrichissement</p>
                  <p className="text-3xl font-bold">Auto</p>
                  <p className="text-green-100 text-sm">Paramètres configurables</p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-200" />
              </div>
            </div>

            {/* Formats supportés */}
            <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm font-medium">Formats supportés</p>
                  <p className="text-3xl font-bold">CSV</p>
                  <p className="text-purple-100 text-sm">Import & Export</p>
                </div>
                <Download className="h-8 w-8 text-purple-200" />
              </div>
            </div>
          </div>
        </div>

        {/* Information Banner */}
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-blue-300 mb-2">Comment utiliser la collecte de données</h4>
                <Badge variant="outline" className="border-blue-500 text-blue-300">
                  Guide
                </Badge>
              </div>
              <div className="text-sm text-blue-200 space-y-1">
                <p>1. <strong>Importez</strong> vos listes CSV existantes avec les colonnes SIREN et nom d'entreprise</p>
                <p>2. <strong>Configurez</strong> les paramètres d'enrichissement selon vos besoins</p>
                <p>3. <strong>Lancez</strong> les sources de scraping pour enrichir automatiquement</p>
                <p>4. <strong>Consultez</strong> la timeline pour suivre les opérations en temps réel</p>
              </div>
            </div>
          </div>
        </div>

        {/* Sources de scraping automatique */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <Zap className="h-6 w-6 text-yellow-400" />
                Sources de scraping automatique
              </h2>
              <p className="text-gray-400 mt-1">
                Collectez automatiquement des données depuis différentes sources
              </p>
            </div>
            
            <Badge variant="outline" className="border-gray-600 text-gray-300">
              3 sources disponibles
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Pappers API */}
            <Card className="bg-gray-800 border-gray-700 p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                    <Database className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Pappers API</h3>
                    <Badge variant="secondary" className="mt-1 bg-green-500/20 text-green-400 border-green-500/30">
                      PRÊT
                    </Badge>
                  </div>
                </div>
              </div>
              <p className="text-gray-400 text-sm mb-4">
                Recherche via l'API Pappers - Données officielles
              </p>
              <div className="space-y-2 mb-4">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full border-blue-500 text-blue-400 hover:bg-blue-500/10"
                  onClick={() => pappersMutation.mutate()}
                  disabled={pappersMutation.isLoading}
                >
                  {pappersMutation.isLoading ? (
                    <RefreshCcw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Lancer
                </Button>
                <Button variant="ghost" size="sm" className="w-full text-gray-400 hover:text-white">
                  <RefreshCcw className="h-4 w-4 mr-2" />
                  Actualiser
                </Button>
              </div>
            </Card>

            {/* Société.com */}
            <Card className="bg-gray-800 border-gray-700 p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
                    <Activity className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Société.com</h3>
                    <Badge variant="secondary" className="mt-1 bg-green-500/20 text-green-400 border-green-500/30">
                      PRÊT
                    </Badge>
                  </div>
                </div>
              </div>
              <p className="text-gray-400 text-sm mb-4">
                Scraping du site Société.com - Données enrichies
              </p>
              <div className="space-y-2 mb-4">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full border-purple-500 text-purple-400 hover:bg-purple-500/10"
                  onClick={() => societeMutation.mutate()}
                  disabled={societeMutation.isLoading}
                >
                  {societeMutation.isLoading ? (
                    <RefreshCcw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Lancer
                </Button>
                <Button variant="ghost" size="sm" className="w-full text-gray-400 hover:text-white">
                  <RefreshCcw className="h-4 w-4 mr-2" />
                  Actualiser
                </Button>
              </div>
            </Card>

            {/* Infogreffe */}
            <Card className="bg-gray-800 border-gray-700 p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                    <FileUp className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Infogreffe</h3>
                    <Badge variant="secondary" className="mt-1 bg-green-500/20 text-green-400 border-green-500/30">
                      PRÊT
                    </Badge>
                  </div>
                </div>
              </div>
              <p className="text-gray-400 text-sm mb-4">
                Enrichissement Infogreffe - Données officielles
              </p>
              <div className="space-y-2 mb-4">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full border-green-500 text-green-400 hover:bg-green-500/10"
                  onClick={() => infogreffeMutation.mutate()}
                  disabled={infogreffeMutation.isLoading}
                >
                  {infogreffeMutation.isLoading ? (
                    <RefreshCcw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Lancer
                </Button>
                <Button variant="ghost" size="sm" className="w-full text-gray-400 hover:text-white">
                  <RefreshCcw className="h-4 w-4 mr-2" />
                  Actualiser
                </Button>
              </div>
            </Card>
          </div>
        </div>

        {/* Activity Timeline */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <Activity className="h-6 w-6 text-blue-400" />
                Activité récente
              </h2>
            </div>
          </div>

          <Card className="bg-gray-800 border-gray-700 p-6">
            <div className="text-center py-12">
              <Activity className="h-12 w-12 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400">Aucune activité récente</p>
              <p className="text-gray-500 text-sm mt-1">
                Les opérations de collecte et d'enrichissement apparaîtront ici
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}