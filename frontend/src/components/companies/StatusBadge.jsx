import React from 'react';
import { Badge } from '../ui/badge';
import { cn } from '../../lib/utils';

const statusConfig = {
  'prospect': {
    variant: 'default',
    className: 'bg-blue-500 hover:bg-blue-600 text-white',
    label: 'Prospect'
  },
  'contact': {
    variant: 'secondary', 
    className: 'bg-yellow-500 hover:bg-yellow-600 text-white',
    label: 'Contact'
  },
  'qualification': {
    variant: 'destructive',
    className: 'bg-purple-500 hover:bg-purple-600 text-white', 
    label: 'Qualification'
  },
  'negociation': {
    variant: 'destructive',
    className: 'bg-orange-500 hover:bg-orange-600 text-white', 
    label: 'Négociation'
  },
  'client': {
    variant: 'success',
    className: 'bg-green-500 hover:bg-green-600 text-white',
    label: 'Client'
  },
  'perdu': {
    variant: 'outline',
    className: 'bg-gray-500 hover:bg-gray-600 text-white',
    label: 'Perdu'
  }
};

export function StatusBadge({ status, className, ...props }) {
  const config = statusConfig[status] || statusConfig['prospect'];
  
  return (
    <Badge
      variant={config.variant}
      className={cn(
        config.className,
        "transition-all duration-200 hover:scale-105 cursor-default",
        className
      )}
      {...props}
    >
      {config.label}
    </Badge>
  );
}