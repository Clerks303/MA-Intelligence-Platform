import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { AlertWithIcon } from '../ui/alert';
import { 
  Play, 
  Square, 
  RefreshCw, 
  Database, 
  Globe, 
  FileText,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { cn } from '../../lib/utils';
import api from '../../services/api';

const sourceIcons = {
  pappers: Database,
  societe: Globe,
  infogreffe: FileText,
};

const sourceColors = {
  pappers: 'blue',
  societe: 'purple', 
  infogreffe: 'green',
};

const statusVariants = {
  ready: { variant: 'secondary', label: 'Prêt', icon: Clock },
  running: { variant: 'default', label: 'En cours', icon: Loader2 },
  completed: { variant: 'success', label: 'Terminé', icon: CheckCircle },
  error: { variant: 'destructive', label: 'Erreur', icon: AlertCircle },
};

export function ScrapingCard({ source, onStart }) {
  const [status, setStatus] = useState(null);
  const [intervalId, setIntervalId] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkStatus = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const response = await api.get(`/scraping/status/${source.id}`);
      setStatus(response.data);
      
      if (!response.data.is_running && intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }
    } catch (error) {
      console.error('Error checking status:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, [source.id, intervalId]);

  useEffect(() => {
    checkStatus();
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [checkStatus, intervalId]);

  const handleStart = async () => {
    try {
      await onStart();
      const id = setInterval(checkStatus, 2000);
      setIntervalId(id);
      checkStatus();
    } catch (error) {
      console.error('Error starting scraping:', error);
    }
  };

  const handleStop = () => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  };

  const getStatusInfo = () => {
    if (!status) return statusVariants.ready;
    
    if (status.is_running) {
      return statusVariants.running;
    } else if (status.error) {
      return statusVariants.error;
    } else if (status.progress === 100) {
      return statusVariants.completed;
    }
    return statusVariants.ready;
  };

  const statusInfo = getStatusInfo();
  const SourceIcon = sourceIcons[source.id] || Database;
  const StatusIcon = statusInfo.icon;
  const colorScheme = sourceColors[source.id] || 'blue';

  const colorClasses = {
    blue: 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950',
    purple: 'border-purple-200 bg-purple-50 dark:border-purple-800 dark:bg-purple-950',
    green: 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950',
  };

  const iconColorClasses = {
    blue: 'text-blue-600 dark:text-blue-400',
    purple: 'text-purple-600 dark:text-purple-400', 
    green: 'text-green-600 dark:text-green-400',
  };

  return (
    <Card className={cn(
      "relative transition-all duration-300 hover:shadow-lg group",
      colorClasses[colorScheme]
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={cn(
              "p-3 rounded-lg bg-white/50 dark:bg-gray-800/50 transition-transform duration-200 group-hover:scale-110",
              iconColorClasses[colorScheme]
            )}>
              <SourceIcon className="h-6 w-6" />
            </div>
            <div>
              <CardTitle className="text-lg">{source.name}</CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                {source.description}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Badge 
              variant={statusInfo.variant}
              className="gap-1"
            >
              <StatusIcon className={cn(
                "h-3 w-3",
                statusInfo.icon === Loader2 && "animate-spin"
              )} />
              {statusInfo.label}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress Section */}
        {status && status.is_running && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{status.message}</span>
              <span className="font-medium">{Math.round(status.progress)}%</span>
            </div>
            
            <Progress value={status.progress} className="h-2" />
            
            {(status.new_companies > 0 || status.skipped_companies > 0) && (
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-green-500" />
                  <span className="text-green-600 dark:text-green-400 font-medium">
                    +{status.new_companies} nouvelles
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-muted-foreground">
                    {status.skipped_companies} ignorées
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Alert */}
        {status && status.error && (
          <AlertWithIcon variant="destructive" title="Erreur lors du scraping">
            {status.error}
          </AlertWithIcon>
        )}

        {/* Success Summary */}
        {status && !status.is_running && !status.error && status.progress === 100 && (
          <AlertWithIcon variant="success" title="Scraping terminé">
            {status.new_companies > 0 && (
              <span>{status.new_companies} nouvelles entreprises ajoutées</span>
            )}
          </AlertWithIcon>
        )}

        {/* Action Buttons */}
        <div className="flex items-center gap-2 pt-2">
          {status && status.is_running ? (
            <Button 
              variant="destructive"
              size="sm"
              onClick={handleStop}
              disabled
              className="gap-2"
            >
              <Square className="h-4 w-4" />
              Arrêter
            </Button>
          ) : (
            <Button 
              size="sm"
              onClick={handleStart}
              disabled={status?.is_running}
              className="gap-2"
            >
              <Play className="h-4 w-4" />
              Lancer
            </Button>
          )}
          
          <Button 
            variant="outline"
            size="sm"
            onClick={checkStatus}
            disabled={isRefreshing}
            className="gap-2"
          >
            <RefreshCw className={cn(
              "h-4 w-4",
              isRefreshing && "animate-spin"
            )} />
            Actualiser
          </Button>
        </div>
      </CardContent>

      {/* Animated border effect */}
      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-out pointer-events-none" />
    </Card>
  );
}