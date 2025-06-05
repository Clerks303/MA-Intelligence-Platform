/**
 * Modern Sidebar Navigation - M&A Intelligence Platform
 * Inspired by the professional UI shown in screenshots
 * Features: dark theme, icons, tooltips, active states
 */

import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard,
  Building2,
  Database,
  Settings,
  LogOut,
  User,
  ChevronDown,
  Bell,
  Moon,
  Sun,
  Menu,
  X
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';

export interface SidebarProps {
  className?: string;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
  href: string;
  badge?: {
    count: number;
    variant: 'default' | 'destructive' | 'warning' | 'success';
  };
  description?: string;
}

const navigationItems: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Tableau de bord',
    icon: LayoutDashboard,
    href: '/dashboard',
    description: 'Vue d\'ensemble de votre pipeline M&A'
  },
  {
    id: 'companies',
    label: 'Entreprises',
    icon: Building2,
    href: '/companies',
    description: 'Gestion des entreprises prospects'
  },
  {
    id: 'scraping',
    label: 'Collecte données',
    icon: Database,
    href: '/scraping',
    description: 'Import et enrichissement automatique'
  },
  {
    id: 'settings',
    label: 'Paramètres',
    icon: Settings,
    href: '/settings',
    description: 'Configuration de la plateforme'
  }
];

export const Sidebar: React.FC<SidebarProps> = ({ 
  className,
  collapsed = false,
  onToggleCollapse
}) => {
  const location = useLocation();
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const isActiveRoute = (href: string) => {
    if (href === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname.startsWith(href);
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <aside 
      className={cn(
        "flex flex-col h-screen transition-all duration-300 ease-in-out",
        "bg-slate-900 border-r border-slate-800 text-white",
        collapsed ? "w-16" : "w-64",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-800">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">MI</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">M&A Intelligence</h1>
              <p className="text-xs text-slate-400">Plateforme de prospection</p>
            </div>
          </div>
        )}
        
        {collapsed && (
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mx-auto">
            <span className="text-white font-bold text-sm">MI</span>
          </div>
        )}

        {onToggleCollapse && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleCollapse}
            className="text-slate-400 hover:text-white hover:bg-slate-800 p-1"
          >
            {collapsed ? <Menu className="h-4 w-4" /> : <X className="h-4 w-4" />}
          </Button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = isActiveRoute(item.href);
          
          return (
            <Link
              key={item.id}
              to={item.href}
              className={cn(
                "nav-item group relative",
                "flex items-center gap-3 px-3 py-2.5 rounded-lg",
                "transition-all duration-200",
                isActive
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-slate-300 hover:bg-slate-800 hover:text-white",
                collapsed && "justify-center"
              )}
              title={collapsed ? item.label : undefined}
            >
              <Icon className={cn(
                "flex-shrink-0 transition-colors",
                isActive ? "text-white" : "text-slate-400 group-hover:text-white",
                "h-5 w-5"
              )} />
              
              {!collapsed && (
                <>
                  <span className="font-medium text-sm">{item.label}</span>
                  {item.badge && (
                    <Badge 
                      variant={item.badge.variant}
                      className="ml-auto h-5 px-2 text-xs"
                    >
                      {item.badge.count}
                    </Badge>
                  )}
                </>
              )}

              {/* Tooltip for collapsed mode */}
              {collapsed && (
                <div className="absolute left-full ml-2 px-3 py-2 bg-slate-800 text-white text-sm rounded-lg opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 whitespace-nowrap border border-slate-700">
                  <div className="font-medium">{item.label}</div>
                  {item.description && (
                    <div className="text-xs text-slate-400 mt-1">{item.description}</div>
                  )}
                </div>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Theme Toggle */}
      <div className="p-4 border-t border-slate-800">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleTheme}
          className={cn(
            "w-full justify-start gap-3 text-slate-300 hover:bg-slate-800 hover:text-white",
            collapsed && "justify-center"
          )}
          title={collapsed ? 'Changer de thème' : undefined}
        >
          {theme === 'dark' ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
          {!collapsed && (
            <span className="text-sm">
              {theme === 'dark' ? 'Mode clair' : 'Mode sombre'}
            </span>
          )}
        </Button>
      </div>

      {/* User Menu */}
      <div className="p-4 border-t border-slate-800">
        <div className="relative">
          <button
            onClick={() => setUserMenuOpen(!userMenuOpen)}
            className={cn(
              "w-full flex items-center gap-3 p-2 rounded-lg",
              "text-slate-300 hover:bg-slate-800 hover:text-white transition-colors",
              collapsed && "justify-center"
            )}
          >
            <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center">
              <User className="h-4 w-4" />
            </div>
            
            {!collapsed && (
              <>
                <div className="flex-1 text-left">
                  <div className="text-sm font-medium text-white">{user?.email || 'admin'}</div>
                  <div className="text-xs text-slate-400">Connecté</div>
                </div>
                <ChevronDown className={cn(
                  "h-4 w-4 transition-transform",
                  userMenuOpen && "rotate-180"
                )} />
              </>
            )}
          </button>

          {/* User Dropdown */}
          {userMenuOpen && !collapsed && (
            <div className="absolute bottom-full left-0 right-0 mb-2 bg-slate-800 border border-slate-700 rounded-lg shadow-lg">
              <div className="p-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                  className="w-full justify-start gap-2 text-slate-300 hover:text-white hover:bg-slate-700"
                >
                  <LogOut className="h-4 w-4" />
                  Déconnexion
                </Button>
              </div>
            </div>
          )}

          {/* Logout button for collapsed mode */}
          {collapsed && (
            <div className="mt-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="w-full justify-center text-slate-300 hover:bg-slate-800 hover:text-white"
                title="Déconnexion"
              >
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Notification badge */}
      <div className="absolute top-4 right-4">
        <Button
          variant="ghost"
          size="sm"
          className="relative text-slate-400 hover:text-white"
        >
          <Bell className="h-4 w-4" />
          <Badge 
            variant="destructive" 
            className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs"
          >
            3
          </Badge>
        </Button>
      </div>
    </aside>
  );
};

export default Sidebar;