/**
 * Widget SLA pour Dashboard - M&A Intelligence Platform
 * Sprint 2 - Indicateurs SLA et qualité des données
 */

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { cn } from '../../../lib/utils';
import { SLAIndicator, DataQualityMetric } from '../types';
// Fonction de formatage de distance de temps simplifiée
const formatDistanceToNow = (date: Date) => {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (diffInSeconds < 60) return 'il y a moins d\'une minute';
  if (diffInSeconds < 3600) return `il y a ${Math.floor(diffInSeconds / 60)} minutes`;
  if (diffInSeconds < 86400) return `il y a ${Math.floor(diffInSeconds / 3600)} heures`;
  return `il y a ${Math.floor(diffInSeconds / 86400)} jours`;
};

interface SLAWidgetProps {
  indicators: SLAIndicator[];
  className?: string;
}

interface DataQualityWidgetProps {
  metrics: DataQualityMetric[];
  className?: string;
}

// Composant pour un indicateur SLA individuel
const SLAIndicatorItem: React.FC<{ indicator: SLAIndicator }> = ({ indicator }) => {
  const getStatusColor = (status: SLAIndicator['status']) => {
    switch (status) {
      case 'excellent':
        return 'text-ma-green-600 bg-ma-green-100';
      case 'good':
        return 'text-ma-green-600 bg-ma-green-50';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'critical':
        return 'text-ma-red-600 bg-ma-red-100';
      default:
        return 'text-ma-slate-600 bg-ma-slate-100';
    }
  };

  const getStatusText = (status: SLAIndicator['status']) => {
    switch (status) {
      case 'excellent':
        return 'Excellent';
      case 'good':
        return 'Bon';
      case 'warning':
        return 'Attention';
      case 'critical':
        return 'Critique';
      default:
        return 'Inconnu';
    }
  };

  const getTrendIcon = (trend: SLAIndicator['trend']) => {
    switch (trend) {
      case 'up':
        return (
          <svg className="w-4 h-4 text-ma-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        );
      case 'down':
        return (
          <svg className="w-4 h-4 text-ma-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        );
    }
  };

  const progressPercentage = Math.min((indicator.current / indicator.target) * 100, 100);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-medium text-ma-slate-900">{indicator.name}</h4>
          <p className="text-xs text-ma-slate-500">
            Objectif: {indicator.target}{indicator.unit}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
            getStatusColor(indicator.status)
          )}>
            {getStatusText(indicator.status)}
          </span>
          {getTrendIcon(indicator.trend)}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-ma-slate-900">
            {indicator.current}{indicator.unit}
          </span>
          <span className="text-ma-slate-500">
            {progressPercentage.toFixed(1)}%
          </span>
        </div>
        
        <div className="w-full bg-ma-slate-200 rounded-full h-2">
          <div
            className={cn(
              "h-2 rounded-full transition-all duration-300",
              indicator.status === 'excellent' || indicator.status === 'good' 
                ? "bg-ma-green-500" 
                : indicator.status === 'warning'
                ? "bg-yellow-500"
                : "bg-ma-red-500"
            )}
            style={{ width: `${Math.min(progressPercentage, 100)}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-ma-slate-500">
        Mis à jour {formatDistanceToNow(indicator.lastUpdate)}
      </p>
    </div>
  );
};

// Widget SLA principal
export const SLAWidget: React.FC<SLAWidgetProps> = ({ indicators, className }) => {
  if (indicators.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Indicateurs SLA</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <svg className="w-12 h-12 text-ma-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-ma-slate-600">Aucun indicateur SLA configuré</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculer le score SLA global
  const globalScore = indicators.reduce((sum, indicator) => {
    return sum + Math.min((indicator.current / indicator.target) * 100, 100);
  }, 0) / indicators.length;

  const criticalCount = indicators.filter(i => i.status === 'critical').length;
  const warningCount = indicators.filter(i => i.status === 'warning').length;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Indicateurs SLA</CardTitle>
          <div className="flex items-center gap-2">
            <span className={cn(
              "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium",
              globalScore >= 95 ? "bg-ma-green-100 text-ma-green-800" :
              globalScore >= 85 ? "bg-yellow-100 text-yellow-800" :
              "bg-ma-red-100 text-ma-red-800"
            )}>
              {globalScore.toFixed(1)}%
            </span>
            {criticalCount > 0 && (
              <span className="bg-ma-red-600 text-white text-xs px-2 py-1 rounded-full font-bold">
                {criticalCount}
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {indicators.map((indicator) => (
          <SLAIndicatorItem key={indicator.id} indicator={indicator} />
        ))}
      </CardContent>
    </Card>
  );
};

// Composant pour métriques de qualité des données
const DataQualityItem: React.FC<{ metric: DataQualityMetric }> = ({ metric }) => {
  const getQualityColor = (score: number) => {
    if (score >= 90) return 'text-ma-green-600 bg-ma-green-100';
    if (score >= 75) return 'text-yellow-600 bg-yellow-100';
    return 'text-ma-red-600 bg-ma-red-100';
  };

  const getQualityText = (score: number) => {
    if (score >= 90) return 'Excellente';
    if (score >= 75) return 'Bonne';
    if (score >= 50) return 'Moyenne';
    return 'Faible';
  };

  const criticalIssues = metric.issues.filter(issue => issue.severity === 'high').length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-medium text-ma-slate-900">{metric.source}</h4>
          <p className="text-xs text-ma-slate-500">
            Données mises à jour il y a {metric.freshness}h
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
            getQualityColor(metric.overall)
          )}>
            {getQualityText(metric.overall)}
          </span>
          {criticalIssues > 0 && (
            <span className="bg-ma-red-600 text-white text-xs px-2 py-1 rounded-full font-bold">
              {criticalIssues}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-ma-slate-600">Complétude</span>
            <span className="font-medium">{metric.completeness}%</span>
          </div>
          <div className="w-full bg-ma-slate-200 rounded-full h-1.5">
            <div
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                getQualityColor(metric.completeness).includes('green') ? "bg-ma-green-500" :
                getQualityColor(metric.completeness).includes('yellow') ? "bg-yellow-500" : "bg-ma-red-500"
              )}
              style={{ width: `${metric.completeness}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-ma-slate-600">Précision</span>
            <span className="font-medium">{metric.accuracy}%</span>
          </div>
          <div className="w-full bg-ma-slate-200 rounded-full h-1.5">
            <div
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                getQualityColor(metric.accuracy).includes('green') ? "bg-ma-green-500" :
                getQualityColor(metric.accuracy).includes('yellow') ? "bg-yellow-500" : "bg-ma-red-500"
              )}
              style={{ width: `${metric.accuracy}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-ma-slate-600">Fraîcheur</span>
            <span className="font-medium">
              {metric.freshness < 24 ? 'Récent' : 'Ancien'}
            </span>
          </div>
          <div className="w-full bg-ma-slate-200 rounded-full h-1.5">
            <div
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                metric.freshness < 24 ? "bg-ma-green-500" :
                metric.freshness < 72 ? "bg-yellow-500" : "bg-ma-red-500"
              )}
              style={{ width: `${Math.max(100 - metric.freshness, 10)}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-ma-slate-600">Cohérence</span>
            <span className="font-medium">{metric.consistency}%</span>
          </div>
          <div className="w-full bg-ma-slate-200 rounded-full h-1.5">
            <div
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                getQualityColor(metric.consistency).includes('green') ? "bg-ma-green-500" :
                getQualityColor(metric.consistency).includes('yellow') ? "bg-yellow-500" : "bg-ma-red-500"
              )}
              style={{ width: `${metric.consistency}%` }}
            />
          </div>
        </div>
      </div>

      {metric.issues.length > 0 && (
        <div className="border-t border-ma-slate-200 pt-3">
          <p className="text-xs font-medium text-ma-slate-600 mb-2">Problèmes identifiés :</p>
          <div className="flex flex-wrap gap-2">
            {metric.issues.map((issue, index) => (
              <span
                key={index}
                className={cn(
                  "inline-flex items-center px-2 py-1 rounded text-xs",
                  issue.severity === 'high' ? "bg-ma-red-100 text-ma-red-800" :
                  issue.severity === 'medium' ? "bg-yellow-100 text-yellow-800" :
                  "bg-ma-slate-100 text-ma-slate-800"
                )}
              >
                {issue.count} {issue.type}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Widget Qualité des données principal
export const DataQualityWidget: React.FC<DataQualityWidgetProps> = ({ metrics, className }) => {
  if (metrics.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Qualité des Données</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <svg className="w-12 h-12 text-ma-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-ma-slate-600">Aucune métrique de qualité disponible</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculer le score global de qualité
  const globalQuality = metrics.reduce((sum, metric) => sum + metric.overall, 0) / metrics.length;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Qualité des Données</CardTitle>
          <span className={cn(
            "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium",
            globalQuality >= 90 ? "bg-ma-green-100 text-ma-green-800" :
            globalQuality >= 75 ? "bg-yellow-100 text-yellow-800" :
            "bg-ma-red-100 text-ma-red-800"
          )}>
            {globalQuality.toFixed(1)}%
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {metrics.map((metric) => (
          <DataQualityItem key={metric.id} metric={metric} />
        ))}
      </CardContent>
    </Card>
  );
};