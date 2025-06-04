/**
 * Widget Alertes pour Dashboard - M&A Intelligence Platform
 * Sprint 2 - Alertes visuelles et badges intelligents
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { cn } from '../../../lib/utils';
import { AlertLevel } from '../types';
// Fonction de formatage de distance de temps simplifiée
const formatDistanceToNow = (date: Date) => {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) return 'il y a moins d\'une minute';
  if (diffInSeconds < 3600) return `il y a ${Math.floor(diffInSeconds / 60)} minutes`;
  if (diffInSeconds < 86400) return `il y a ${Math.floor(diffInSeconds / 3600)} heures`;
  return `il y a ${Math.floor(diffInSeconds / 86400)} jours`;
};

interface AlertsWidgetProps {
  alerts: AlertLevel[];
  onDismissAlert?: (alertId: string) => void;
  onViewAlert?: (alertId: string) => void;
  className?: string;
  maxVisible?: number;
}

const getAlertIcon = (level: AlertLevel['level']) => {
  switch (level) {
    case 'critical':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'warning':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
      );
    case 'info':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'success':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    default:
      return null;
  }
};

const getAlertStyles = (level: AlertLevel['level']) => {
  switch (level) {
    case 'critical':
      return {
        container: 'border-l-4 border-ma-red-500 bg-ma-red-50',
        icon: 'text-ma-red-600 bg-ma-red-100',
        title: 'text-ma-red-900',
        text: 'text-ma-red-700',
        badge: 'bg-ma-red-600 text-white',
      };
    case 'warning':
      return {
        container: 'border-l-4 border-yellow-500 bg-yellow-50',
        icon: 'text-yellow-600 bg-yellow-100',
        title: 'text-yellow-900',
        text: 'text-yellow-700',
        badge: 'bg-yellow-600 text-white',
      };
    case 'info':
      return {
        container: 'border-l-4 border-ma-blue-500 bg-ma-blue-50',
        icon: 'text-ma-blue-600 bg-ma-blue-100',
        title: 'text-ma-blue-900',
        text: 'text-ma-blue-700',
        badge: 'bg-ma-blue-600 text-white',
      };
    case 'success':
      return {
        container: 'border-l-4 border-ma-green-500 bg-ma-green-50',
        icon: 'text-ma-green-600 bg-ma-green-100',
        title: 'text-ma-green-900',
        text: 'text-ma-green-700',
        badge: 'bg-ma-green-600 text-white',
      };
    default:
      return {
        container: 'border-l-4 border-ma-slate-500 bg-ma-slate-50',
        icon: 'text-ma-slate-600 bg-ma-slate-100',
        title: 'text-ma-slate-900',
        text: 'text-ma-slate-700',
        badge: 'bg-ma-slate-600 text-white',
      };
  }
};

const AlertItem: React.FC<{
  alert: AlertLevel;
  onDismiss?: (id: string) => void;
  onView?: (id: string) => void;
}> = ({ alert, onDismiss, onView }) => {
  const styles = getAlertStyles(alert.level);
  const [isDismissing, setIsDismissing] = useState(false);

  const handleDismiss = async () => {
    if (onDismiss) {
      setIsDismissing(true);
      try {
        await onDismiss(alert.id);
      } catch (error) {
        setIsDismissing(false);
      }
    }
  };

  if (alert.dismissed) {
    return null;
  }

  return (
    <div className={cn(
      "relative p-4 rounded-lg transition-all duration-300",
      styles.container,
      isDismissing && "opacity-50 scale-95"
    )}>
      <div className="flex items-start gap-3">
        <div className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
          styles.icon
        )}>
          {getAlertIcon(alert.level)}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className={cn("text-sm font-medium", styles.title)}>
                {alert.message}
              </p>
              <p className={cn("text-xs mt-1", styles.text)}>
                {formatDistanceToNow(alert.timestamp)}
              </p>
            </div>
            
            <div className="flex items-center gap-2 ml-4">
              {alert.action && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => onView?.(alert.id)}
                  className={cn("text-xs", styles.text)}
                >
                  {alert.action.label}
                </Button>
              )}
              
              {onDismiss && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleDismiss}
                  disabled={isDismissing}
                  className="h-6 w-6 text-ma-slate-400 hover:text-ma-slate-600"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const AlertsWidget: React.FC<AlertsWidgetProps> = ({
  alerts,
  onDismissAlert,
  onViewAlert,
  className,
  maxVisible = 5,
}) => {
  const [showAll, setShowAll] = useState(false);
  
  // Trier les alertes par priorité et date
  const sortedAlerts = [...alerts].sort((a, b) => {
    const priorityOrder = { critical: 4, warning: 3, info: 2, success: 1 };
    const priorityDiff = priorityOrder[b.level] - priorityOrder[a.level];
    if (priorityDiff !== 0) return priorityDiff;
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  const visibleAlerts = showAll ? sortedAlerts : sortedAlerts.slice(0, maxVisible);
  const hasMoreAlerts = sortedAlerts.length > maxVisible;

  // Compter les alertes par type
  const alertCounts = alerts.reduce((acc, alert) => {
    if (!alert.dismissed) {
      acc[alert.level] = (acc[alert.level] || 0) + 1;
    }
    return acc;
  }, {} as Record<AlertLevel['level'], number>);

  if (alerts.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <svg className="w-5 h-5 text-ma-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Alertes Système
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <svg className="w-12 h-12 text-ma-green-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-ma-slate-600 font-medium">Aucune alerte active</p>
            <p className="text-sm text-ma-slate-500 mt-1">Tous les systèmes fonctionnent normalement</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <svg className="w-5 h-5 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5zM15 17H9a4 4 0 01-4-4V5a4 4 0 014-4h6a4 4 0 014 4v8a4 4 0 01-4 4z" />
            </svg>
            Alertes Système
            {alerts.length > 0 && (
              <span className="bg-ma-slate-100 text-ma-slate-700 text-xs px-2 py-1 rounded-full">
                {alerts.filter(a => !a.dismissed).length}
              </span>
            )}
          </CardTitle>
          
          {/* Badges de résumé */}
          <div className="flex gap-2">
            {Object.entries(alertCounts).map(([level, count]) => {
              const styles = getAlertStyles(level as AlertLevel['level']);
              return count > 0 ? (
                <span
                  key={level}
                  className={cn(
                    "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
                    styles.badge
                  )}
                >
                  {count}
                </span>
              ) : null;
            })}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-3">
        {visibleAlerts.map((alert) => (
          <AlertItem
            key={alert.id}
            alert={alert}
            onDismiss={onDismissAlert}
            onView={onViewAlert}
          />
        ))}
        
        {hasMoreAlerts && (
          <div className="pt-3 border-t border-ma-slate-200">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAll(!showAll)}
              className="w-full"
            >
              {showAll 
                ? 'Voir moins' 
                : `Voir ${sortedAlerts.length - maxVisible} alertes supplémentaires`
              }
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Badge intelligent pour le nombre d'alertes (pour la navigation)
interface AlertsBadgeProps {
  alerts: AlertLevel[];
  className?: string;
}

export const AlertsBadge: React.FC<AlertsBadgeProps> = ({ alerts, className }) => {
  const activeAlerts = alerts.filter(alert => !alert.dismissed);
  const criticalCount = activeAlerts.filter(alert => alert.level === 'critical').length;
  const warningCount = activeAlerts.filter(alert => alert.level === 'warning').length;
  
  if (activeAlerts.length === 0) {
    return null;
  }

  return (
    <div className={cn("flex items-center gap-1", className)}>
      {criticalCount > 0 && (
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-bold bg-ma-red-600 text-white">
          {criticalCount}
        </span>
      )}
      {warningCount > 0 && criticalCount === 0 && (
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-bold bg-yellow-600 text-white">
          {warningCount}
        </span>
      )}
    </div>
  );
};