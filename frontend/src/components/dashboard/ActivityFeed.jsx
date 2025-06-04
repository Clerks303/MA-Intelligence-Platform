import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Clock, Activity, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { cn } from '../../lib/utils';

const activityIcons = {
  info: Info,
  success: CheckCircle,
  warning: AlertCircle,
  default: Activity,
};

const activityColors = {
  info: 'info',
  success: 'success', 
  warning: 'warning',
  default: 'secondary',
};

export function ActivityFeed({ activities = [] }) {
  const defaultActivities = [
    { 
      time: 'Il y a 2 min', 
      action: 'Nouveau scraping Pappers lancé', 
      type: 'info',
      details: '127 nouvelles entreprises trouvées'
    },
    { 
      time: 'Il y a 15 min', 
      action: 'Entreprise KPMG mise à jour', 
      type: 'success',
      details: 'Données financières enrichies'
    },
    { 
      time: 'Il y a 1h', 
      action: 'Export CSV généré', 
      type: 'default',
      details: '250 entreprises exportées'
    },
    { 
      time: 'Il y a 2h', 
      action: 'Enrichissement Infogreffe terminé', 
      type: 'success',
      details: '98% de réussite'
    },
    { 
      time: 'Il y a 3h', 
      action: 'Limite API Pappers atteinte', 
      type: 'warning',
      details: 'Renouvellement dans 2h'
    },
  ];

  const displayActivities = activities.length > 0 ? activities : defaultActivities;

  return (
    <Card className="transition-all duration-300 hover:shadow-lg">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-xl font-semibold">
          <Activity className="h-5 w-5 text-primary" />
          Activité récente
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-4">
          {displayActivities.map((activity, index) => {
            const IconComponent = activityIcons[activity.type] || activityIcons.default;
            
            return (
              <div 
                key={index} 
                className={cn(
                  "flex items-start gap-4 p-3 rounded-lg transition-all duration-200",
                  "hover:bg-muted/50 group cursor-pointer",
                  "animate-fade-in"
                )}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={cn(
                  "flex-shrink-0 p-2 rounded-full transition-transform duration-200 group-hover:scale-110",
                  activity.type === 'info' && "bg-blue-100 text-blue-600 dark:bg-blue-950 dark:text-blue-400",
                  activity.type === 'success' && "bg-green-100 text-green-600 dark:bg-green-950 dark:text-green-400",
                  activity.type === 'warning' && "bg-yellow-100 text-yellow-600 dark:bg-yellow-950 dark:text-yellow-400",
                  activity.type === 'default' && "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
                )}>
                  <IconComponent className="h-4 w-4" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">
                      {activity.action}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {activity.time}
                    </div>
                  </div>
                  
                  {activity.details && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {activity.details}
                    </p>
                  )}
                  
                  <Badge 
                    variant={activityColors[activity.type]} 
                    className="mt-2 text-xs"
                  >
                    {activity.type === 'info' && 'Information'}
                    {activity.type === 'success' && 'Succès'}
                    {activity.type === 'warning' && 'Attention'}
                    {activity.type === 'default' && 'Activité'}
                  </Badge>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}