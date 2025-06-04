import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { cn } from '../../lib/utils';

export function ChartCard({ 
  title, 
  children, 
  className,
  headerAction,
  description 
}) {
  return (
    <Card className={cn(
      "transition-all duration-300 hover:shadow-lg",
      className
    )}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-semibold">{title}</CardTitle>
            {description && (
              <p className="text-sm text-muted-foreground mt-1">{description}</p>
            )}
          </div>
          {headerAction && (
            <div className="flex items-center space-x-2">
              {headerAction}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-[350px] w-full">
          {children}
        </div>
      </CardContent>
    </Card>
  );
}