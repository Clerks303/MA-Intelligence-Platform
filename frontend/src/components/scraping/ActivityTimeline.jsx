import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { 
  Activity,
  CheckCircle,
  AlertCircle,
  Info,
  Clock,
  Upload,
  Download,
  Zap,
  TrendingUp
} from 'lucide-react';
import { cn } from '../../lib/utils';

const activityIcons = {
  success: CheckCircle,
  error: AlertCircle,
  info: Info,
  upload: Upload,
  download: Download,
  scraping: Zap,
  enrichment: TrendingUp,
};

const activityColors = {
  success: 'text-green-500 bg-green-100 dark:bg-green-900',
  error: 'text-red-500 bg-red-100 dark:bg-red-900',
  info: 'text-blue-500 bg-blue-100 dark:bg-blue-900',
  upload: 'text-purple-500 bg-purple-100 dark:bg-purple-900',
  download: 'text-indigo-500 bg-indigo-100 dark:bg-indigo-900',
  scraping: 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900',
  enrichment: 'text-orange-500 bg-orange-100 dark:bg-orange-900',
};

const badgeVariants = {
  success: 'success',
  error: 'destructive',
  info: 'info',
  upload: 'secondary',
  download: 'secondary',
  scraping: 'warning',
  enrichment: 'info',
};

export function ActivityTimeline({ activities = [] }) {
  // Mock data for demonstration
  const defaultActivities = [
    {
      id: 1,
      type: 'success',
      title: 'Import CSV terminé',
      description: '150 nouvelles entreprises ajoutées',
      timestamp: 'il y a 5 minutes',
      details: { companies: 150, source: 'upload' }
    },
    {
      id: 2,
      type: 'scraping',
      title: 'Scraping Pappers en cours',
      description: 'Département 75 - 45% complété',
      timestamp: 'il y a 10 minutes',
      details: { progress: 45, found: 89, department: '75' }
    },
    {
      id: 3,
      type: 'enrichment',
      title: 'Enrichissement Infogreffe terminé',
      description: '234 entreprises enrichies avec succès',
      timestamp: 'il y a 1 heure',
      details: { enriched: 234, failed: 12 }
    },
    {
      id: 4,
      type: 'error',
      title: 'Erreur Société.com',
      description: 'Captcha détecté, relance nécessaire',
      timestamp: 'il y a 2 heures',
      details: { error: 'Captcha protection activated' }
    },
    {
      id: 5,
      type: 'download',
      title: 'Export CSV généré',
      description: '342 entreprises exportées',
      timestamp: 'il y a 3 heures',
      details: { exported: 342, format: 'CSV' }
    },
    {
      id: 6,
      type: 'info',
      title: 'Démarrage du système',
      description: 'Initialisation des services de scraping',
      timestamp: 'il y a 4 heures',
      details: { services: ['pappers', 'societe', 'infogreffe'] }
    }
  ];

  const displayActivities = activities.length > 0 ? activities : defaultActivities;

  return (
    <Card className="transition-all duration-300 hover:shadow-lg">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-xl font-semibold">
          <Activity className="h-5 w-5 text-primary" />
          Timeline d'activité
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Historique des dernières opérations de scraping et d'enrichissement
        </p>
      </CardHeader>
      
      <CardContent className="pt-0">
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-6 top-0 bottom-0 w-px bg-border" />
          
          <div className="space-y-6">
            {displayActivities.map((activity, index) => {
              if (!activity) return null;
              const IconComponent = activityIcons[activity?.type] || Info;
              const colorClass = activityColors[activity?.type] || activityColors.info;
              const badgeVariant = badgeVariants[activity?.type] || 'secondary';
              
              return (
                <div 
                  key={activity.id} 
                  className={cn(
                    "relative flex items-start gap-4 transition-all duration-200",
                    "hover:bg-muted/30 -mx-2 px-2 py-2 rounded-lg group cursor-pointer",
                    "animate-fade-in"
                  )}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  {/* Icon */}
                  <div className={cn(
                    "relative z-10 flex-shrink-0 p-2 rounded-full transition-all duration-200",
                    "group-hover:scale-110 group-hover:shadow-md",
                    colorClass
                  )}>
                    <IconComponent className="h-4 w-4" />
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <h4 className="font-medium text-foreground group-hover:text-primary transition-colors">
                        {activity?.title || 'Activité sans titre'}
                      </h4>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span>{activity?.timestamp || 'Date inconnue'}</span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-2">
                      {activity?.description || 'Aucune description'}
                    </p>
                    
                    <div className="flex items-center gap-2">
                      <Badge variant={badgeVariant} className="text-xs">
                        {activity?.type === 'success' && 'Succès'}
                        {activity?.type === 'error' && 'Erreur'}
                        {activity?.type === 'info' && 'Information'}
                        {activity?.type === 'upload' && 'Import'}
                        {activity?.type === 'download' && 'Export'}
                        {activity?.type === 'scraping' && 'Scraping'}
                        {activity?.type === 'enrichment' && 'Enrichissement'}
                      </Badge>
                      
                      {/* Additional details badges */}
                      {activity?.details && (
                        <>
                          {activity?.details?.companies && (
                            <Badge variant="outline" className="text-xs">
                              {activity.details.companies} entreprises
                            </Badge>
                          )}
                          {activity?.details?.progress && (
                            <Badge variant="outline" className="text-xs">
                              {activity.details.progress}% complété
                            </Badge>
                          )}
                          {activity?.details?.found && (
                            <Badge variant="outline" className="text-xs">
                              {activity.details.found} trouvées
                            </Badge>
                          )}
                          {activity?.details?.enriched && (
                            <Badge variant="outline" className="text-xs">
                              {activity.details.enriched} enrichies
                            </Badge>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {displayActivities.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Aucune activité récente</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}