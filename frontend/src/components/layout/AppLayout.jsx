/**
 * Modern App Layout - M&A Intelligence Platform
 * Main layout component with sidebar navigation and content area
 * Inspired by professional UI patterns from screenshots
 */

import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { cn } from '../../lib/utils';
import { Sidebar } from '../navigation/Sidebar';
import { useLocalStorage } from '../../hooks/useLocalStorage';
import { Bell, Search, Settings } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';

export interface AppLayoutProps {
  className?: string;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ className }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useLocalStorage('sidebar-collapsed', false);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update time every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 60000);

    return () => clearInterval(interval);
  }, []);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className={cn("flex h-screen bg-slate-50 dark:bg-slate-900", className)}>
      {/* Sidebar */}
      <Sidebar 
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleSidebar}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header Bar */}
        <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Page Title Area */}
            <div className="flex items-center gap-4">
              <div>
                <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
                  Tableau de bord
                </h1>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Vue d'ensemble de votre pipeline M&A
                </p>
              </div>
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center gap-4">
              {/* Search */}
              <div className="relative hidden md:block">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <Input
                  placeholder="Rechercher une entreprise..."
                  className="pl-10 w-64 bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700"
                />
              </div>

              {/* Current Date */}
              <div className="hidden lg:flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <span>Mise à jour:</span>
                <span className="font-medium">
                  {currentTime.toLocaleDateString('fr-FR', {
                    day: '2-digit',
                    month: '2-digit',
                    year: 'numeric'
                  })}
                </span>
              </div>

              {/* Refresh Button */}
              <Button
                variant="outline"
                size="sm"
                className="gap-2 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <Settings className="h-4 w-4" />
                <span className="hidden sm:inline">Actualiser</span>
              </Button>

              {/* Notifications */}
              <div className="relative">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                >
                  <Bell className="h-5 w-5" />
                </Button>
                <Badge 
                  variant="destructive" 
                  className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs"
                >
                  3
                </Badge>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-6">
            <Outlet />
          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 px-6 py-3">
          <div className="flex items-center justify-between text-sm text-slate-500 dark:text-slate-400">
            <div className="flex items-center gap-4">
              <span>© 2025 M&A Intelligence Platform</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">Version 2.0</span>
            </div>
            
            <div className="flex items-center gap-4">
              <span className="hidden md:inline">
                Dernière synchronisation: {currentTime.toLocaleTimeString('fr-FR', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}
              </span>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs">En ligne</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default AppLayout;