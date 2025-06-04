import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  Menu as MenuIcon, 
  LayoutDashboard, 
  Building2, 
  Download, 
  Settings, 
  LogOut, 
  Bell, 
  User, 
  X,
  Home,
  Activity,
  FileText
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { ThemeToggle } from './ui/theme-toggle';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';

const menuItems = [
  { text: 'Tableau de bord', icon: LayoutDashboard, path: '/dashboard' },
  { text: 'Analytics Avancé', icon: Activity, path: '/analytics' },
  { text: 'Monitoring', icon: Activity, path: '/monitoring' },
  { text: 'Entreprises', icon: Building2, path: '/companies' },
  { text: 'Collecte données', icon: Download, path: '/scraping' },
  { text: 'Documents', icon: FileText, path: '/documents' },
  { text: 'Édition collaborative', icon: FileText, path: '/collaborative' },
  { text: 'Paramètres', icon: Settings, path: '/settings' },
];

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Sidebar Overlay for Mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        "fixed left-0 top-0 h-full w-64 bg-card border-r transform transition-transform duration-300 ease-in-out z-50",
        "lg:relative lg:translate-x-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="p-6 border-b">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary text-primary-foreground">
                <Building2 className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">
                  M&A Intelligence
                </h1>
                <p className="text-xs text-muted-foreground">
                  Plateforme de prospection
                </p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4">
            <ul className="space-y-2">
              {menuItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path || 
                  (item.path === '/documents' && location.pathname.startsWith('/documents')) ||
                  (item.path === '/collaborative' && location.pathname.startsWith('/collaborative')) ||
                  (item.path === '/analytics' && location.pathname.startsWith('/analytics'));
                
                return (
                  <li key={item.path}>
                    <button
                      onClick={() => {
                        navigate(item.path);
                        setSidebarOpen(false);
                      }}
                      className={cn(
                        "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all duration-200",
                        "hover:bg-accent hover:text-accent-foreground",
                        isActive 
                          ? "bg-primary text-primary-foreground shadow-sm" 
                          : "text-muted-foreground"
                      )}
                    >
                      <Icon className="h-5 w-5" />
                      <span className="font-medium">{item.text}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>

          {/* User Section */}
          <div className="p-4 border-t">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-full bg-muted">
                <User className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {user?.username || 'Utilisateur'}
                </p>
                <p className="text-xs text-muted-foreground">
                  Connecté
                </p>
              </div>
            </div>
            
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleLogout}
              className="w-full gap-2"
            >
              <LogOut className="h-4 w-4" />
              Déconnexion
            </Button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:ml-0">
        {/* Header */}
        <header className="bg-card border-b px-4 lg:px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden"
            >
              <MenuIcon className="h-5 w-5" />
            </Button>

            {/* Page Title */}
            <div className="hidden lg:block">
              <h2 className="text-xl font-semibold text-foreground">
                {menuItems.find(item => 
                  item.path === location.pathname || 
                  (item.path === '/documents' && location.pathname.startsWith('/documents')) ||
                  (item.path === '/collaborative' && location.pathname.startsWith('/collaborative')) ||
                  (item.path === '/analytics' && location.pathname.startsWith('/analytics'))
                )?.text || 'Dashboard'}
              </h2>
            </div>

            {/* Header Actions */}
            <div className="flex items-center gap-2">
              {/* Notifications */}
              <div className="relative">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setNotificationsOpen(!notificationsOpen)}
                  className="relative"
                >
                  <Bell className="h-5 w-5" />
                  <Badge 
                    className="absolute -top-1 -right-1 h-5 w-5 text-xs p-0 flex items-center justify-center"
                    variant="destructive"
                  >
                    3
                  </Badge>
                </Button>

                {/* Notifications Dropdown */}
                {notificationsOpen && (
                  <div className="absolute right-0 top-full mt-2 w-80 bg-card border rounded-lg shadow-lg z-50">
                    <div className="p-4 border-b">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">Notifications</h3>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => setNotificationsOpen(false)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    
                    <div className="max-h-64 overflow-y-auto">
                      <div className="p-3 border-b hover:bg-muted/50 cursor-pointer">
                        <p className="text-sm font-medium">Nouveau scraping terminé</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          150 nouvelles entreprises ajoutées - il y a 5 min
                        </p>
                      </div>
                      <div className="p-3 border-b hover:bg-muted/50 cursor-pointer">
                        <p className="text-sm font-medium">Export CSV prêt</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Votre export de 500 entreprises est disponible - il y a 1h
                        </p>
                      </div>
                      <div className="p-3 hover:bg-muted/50 cursor-pointer">
                        <p className="text-sm font-medium">Mise à jour système</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Nouvelles fonctionnalités disponibles - il y a 2h
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Theme Toggle */}
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 p-4 lg:p-6 bg-background animate-fade-in">
          <Outlet />
        </main>
      </div>
    </div>
  );
}