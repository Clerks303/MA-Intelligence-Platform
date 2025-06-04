import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthForm } from '../components/auth/AuthForm';
import { ThemeToggle } from '../components/ui/theme-toggle';
import { Badge } from '../components/ui/badge';
import { 
  Building2, 
  TrendingUp, 
  Users, 
  Database,
  Sparkles,
  Shield,
  Zap
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { cn } from '../lib/utils';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentFeature, setCurrentFeature] = useState(0);

  // Features carousel
  const features = [
    {
      icon: Building2,
      title: "Base de données entreprises",
      description: "Accédez à des milliers d'entreprises avec données enrichies"
    },
    {
      icon: TrendingUp,
      title: "Scoring IA avancé",
      description: "Algorithmes d'IA pour évaluer le potentiel M&A"
    },
    {
      icon: Database,
      title: "Scraping automatique",
      description: "Collecte automatisée depuis Pappers, Infogreffe, Société.com"
    },
    {
      icon: Users,
      title: "Pipeline de prospection",
      description: "Gérez vos prospects et suivez vos deals en temps réel"
    }
  ];

  // Auto-rotate features
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [features.length]);

  const handleLogin = async (formData) => {
    setIsLoading(true);
    setError('');
    
    try {
      console.log('🔐 Initiating login process');
      const result = await login(formData.username, formData.password);
      
      console.log('✅ Login successful, redirecting to dashboard');
      // Small delay to show success state
      setTimeout(() => {
        navigate('/dashboard', { replace: true });
      }, 500);
      
    } catch (err) {
      console.error('❌ Login error:', err);
      setError(err.message || 'Identifiants incorrects. Vérifiez votre nom d\'utilisateur et mot de passe.');
    } finally {
      setIsLoading(false);
    }
  };

  const currentFeatureData = features[currentFeature];
  const FeatureIcon = currentFeatureData.icon;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Main Container - Perfectly Centered */}
      <div className="min-h-screen flex items-center justify-center p-4 lg:p-8">
        <div className="w-full max-w-7xl mx-auto">
          
          {/* Desktop Layout */}
          <div className="hidden lg:flex lg:items-center lg:gap-16 xl:gap-24">
            
            {/* Left Side - Branding & Features */}
            <div className="flex-1 max-w-2xl relative">
              {/* Background Pattern */}
              <div className="absolute inset-0 opacity-5 pointer-events-none">
                <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%239C92AC%22%20fill-opacity%3D%220.1%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%224%22/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')]" />
              </div>
              
              <div className="relative z-10 py-12">
                {/* Logo & Brand */}
                <div className="mb-12">
                  <div className="flex items-center gap-4 mb-8">
                    <div className="p-3 rounded-xl bg-primary text-primary-foreground shadow-lg">
                      <Building2 className="h-8 w-8" />
                    </div>
                    <div>
                      <h1 className="text-2xl lg:text-3xl font-bold text-foreground">
                        M&A Intelligence Platform
                      </h1>
                      <p className="text-sm text-muted-foreground">
                        Version 2.0 - Plateforme de prospection M&A
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h2 className="text-3xl lg:text-4xl xl:text-5xl font-bold text-foreground leading-tight">
                      Identifiez les 
                      <span className="text-primary"> opportunités M&A</span> 
                      les plus prometteuses
                    </h2>
                    <p className="text-lg lg:text-xl text-muted-foreground max-w-xl">
                      Plateforme intelligente de prospection pour cabinets comptables. 
                      Collectez, enrichissez et qualifiez automatiquement vos prospects.
                    </p>
                  </div>
                </div>

                {/* Rotating Feature */}
                <div className="mb-10">
                  <div className={cn(
                    "p-6 lg:p-8 rounded-xl border bg-card/50 backdrop-blur-sm transition-all duration-500",
                    "hover:shadow-lg hover:shadow-primary/5 group"
                  )}>
                    <div className="flex items-start gap-4">
                      <div className="p-3 lg:p-4 rounded-lg bg-primary/10 text-primary group-hover:scale-110 transition-transform duration-300">
                        <FeatureIcon className="h-6 w-6 lg:h-7 lg:w-7" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-foreground mb-2 text-lg">
                          {currentFeatureData.title}
                        </h3>
                        <p className="text-muted-foreground">
                          {currentFeatureData.description}
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Feature Indicators */}
                  <div className="flex justify-center gap-2 mt-6">
                    {features.map((_, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentFeature(index)}
                        className={cn(
                          "h-2 rounded-full transition-all duration-300",
                          index === currentFeature 
                            ? "bg-primary w-8" 
                            : "bg-muted-foreground/30 hover:bg-muted-foreground/50 w-2"
                        )}
                      />
                    ))}
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-6 lg:gap-8 mb-8">
                  <div className="text-center">
                    <div className="text-2xl lg:text-3xl font-bold text-primary">10K+</div>
                    <div className="text-sm text-muted-foreground">Entreprises</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl lg:text-3xl font-bold text-primary">95%</div>
                    <div className="text-sm text-muted-foreground">Précision IA</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl lg:text-3xl font-bold text-primary">24/7</div>
                    <div className="text-sm text-muted-foreground">Monitoring</div>
                  </div>
                </div>

                {/* Trust Badges */}
                <div className="flex flex-wrap gap-3">
                  <Badge variant="outline" className="gap-2 px-3 py-1">
                    <Shield className="h-3 w-3" />
                    Sécurisé
                  </Badge>
                  <Badge variant="outline" className="gap-2 px-3 py-1">
                    <Zap className="h-3 w-3" />
                    Temps réel
                  </Badge>
                  <Badge variant="outline" className="gap-2 px-3 py-1">
                    <Sparkles className="h-3 w-3" />
                    IA Avancée
                  </Badge>
                </div>
              </div>
            </div>

            {/* Right Side - Login Form */}
            <div className="w-full max-w-md xl:max-w-lg">
              <div className="bg-card rounded-2xl border shadow-xl p-8 lg:p-10">
                {/* Header */}
                <div className="flex justify-between items-center mb-8">
                  <div>
                    <h3 className="text-2xl font-bold text-foreground">Connexion</h3>
                    <p className="text-muted-foreground mt-1">Accédez à votre plateforme M&A</p>
                  </div>
                  <ThemeToggle />
                </div>

                {/* Login Form */}
                <AuthForm 
                  onSubmit={handleLogin}
                  isLoading={isLoading}
                  error={error}
                  className="w-full"
                />

                {/* Footer */}
                <div className="mt-8 pt-6 border-t text-center">
                  <p className="text-xs text-muted-foreground mb-3">
                    © 2025 M&A Intelligence Platform
                  </p>
                  <div className="flex justify-center gap-4">
                    <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                      Aide
                    </button>
                    <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                      Contact
                    </button>
                    <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                      Confidentialité
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Mobile Layout */}
          <div className="lg:hidden">
            {/* Mobile Header */}
            <div className="flex justify-between items-center mb-8">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary text-primary-foreground">
                  <Building2 className="h-6 w-6" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-foreground">M&A Intelligence</h1>
                  <p className="text-xs text-muted-foreground">Plateforme de prospection</p>
                </div>
              </div>
              <ThemeToggle />
            </div>

            {/* Mobile Hero */}
            <div className="text-center mb-12 px-4">
              <h2 className="text-2xl sm:text-3xl font-bold text-foreground leading-tight mb-4">
                Identifiez les 
                <span className="text-primary"> opportunités M&A</span> 
                prometteuses
              </h2>
              <p className="text-muted-foreground mb-8">
                Plateforme intelligente de prospection pour cabinets comptables.
              </p>
              
              {/* Mobile Stats */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-xl font-bold text-primary">10K+</div>
                  <div className="text-xs text-muted-foreground">Entreprises</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-primary">95%</div>
                  <div className="text-xs text-muted-foreground">Précision</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-primary">24/7</div>
                  <div className="text-xs text-muted-foreground">Support</div>
                </div>
              </div>
            </div>

            {/* Mobile Login Form */}
            <div className="bg-card rounded-2xl border shadow-lg p-6 sm:p-8 mx-auto max-w-md">
              <div className="mb-6">
                <h3 className="text-xl font-bold text-foreground mb-2">Connexion</h3>
                <p className="text-sm text-muted-foreground">Accédez à votre plateforme</p>
              </div>

              <AuthForm 
                onSubmit={handleLogin}
                isLoading={isLoading}
                error={error}
                className="w-full"
              />

              {/* Mobile Footer */}
              <div className="mt-8 pt-6 border-t text-center">
                <p className="text-xs text-muted-foreground mb-3">
                  © 2025 M&A Intelligence Platform
                </p>
                <div className="flex justify-center gap-4">
                  <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                    Aide
                  </button>
                  <button className="text-xs text-muted-foreground hover:text-foreground transition-colors">
                    Contact
                  </button>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}