import React from 'react';
import { Card, CardContent } from '../ui/card';
import { cn } from '../../lib/utils';

export function StatCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  trend,
  trendValue,
  color = "blue",
  className 
}) {
  const colorClasses = {
    blue: "text-blue-600 bg-blue-50 border-blue-200 dark:text-blue-400 dark:bg-blue-950 dark:border-blue-800",
    green: "text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-950 dark:border-green-800",
    yellow: "text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-950 dark:border-yellow-800",
    red: "text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-950 dark:border-red-800",
    purple: "text-purple-600 bg-purple-50 border-purple-200 dark:text-purple-400 dark:bg-purple-950 dark:border-purple-800",
    indigo: "text-indigo-600 bg-indigo-50 border-indigo-200 dark:text-indigo-400 dark:bg-indigo-950 dark:border-indigo-800",
  };

  const iconColorClasses = {
    blue: "text-blue-500",
    green: "text-green-500", 
    yellow: "text-yellow-500",
    red: "text-red-500",
    purple: "text-purple-500",
    indigo: "text-indigo-500",
  };

  return (
    <Card className={cn(
      "relative overflow-hidden border transition-all duration-300 hover:shadow-lg hover:-translate-y-1 group cursor-pointer",
      colorClasses[color],
      className
    )}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2 flex-1">
            <p className="text-sm font-medium uppercase tracking-wide opacity-80">
              {title}
            </p>
            <p className="text-3xl font-bold tracking-tight">
              {value}
            </p>
            {subtitle && (
              <p className="text-sm opacity-70">
                {subtitle}
              </p>
            )}
            {trend && trendValue && (
              <div className={cn(
                "flex items-center gap-1 text-xs font-medium",
                trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}>
                <span className={cn(
                  "inline-flex items-center",
                  trend === 'up' ? '↗' : '↘'
                )}>
                  {trend === 'up' ? '↗' : '↘'}
                </span>
                {trendValue}
              </div>
            )}
          </div>
          
          <div className={cn(
            "p-3 rounded-lg transition-all duration-300 group-hover:scale-110",
            `bg-white/20 backdrop-blur-sm`,
            iconColorClasses[color]
          )}>
            <Icon className="h-6 w-6" />
          </div>
        </div>
        
        {/* Gradient overlay animation on hover */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-out" />
      </CardContent>
    </Card>
  );
}