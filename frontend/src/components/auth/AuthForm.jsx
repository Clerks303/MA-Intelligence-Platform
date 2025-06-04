import React, { useState } from 'react';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Button } from '../ui/button';
import { AlertWithIcon } from '../ui/alert';
import { 
  Eye, 
  EyeOff, 
  User, 
  Lock, 
  Loader2,
  KeyRound,
  LogIn
} from 'lucide-react';
import { cn } from '../../lib/utils';

export function AuthForm({ 
  onSubmit, 
  isLoading = false, 
  error = '',
  className 
}) {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [touched, setTouched] = useState({});
  const [fieldErrors, setFieldErrors] = useState({});

  // Real-time validation
  const validateField = (name, value) => {
    const errors = {};
    
    if (name === 'username') {
      if (!value.trim()) {
        errors.username = 'Nom d\'utilisateur requis';
      } else if (value.length < 2) {
        errors.username = 'Au moins 2 caractères requis';
      }
    }
    
    if (name === 'password') {
      if (!value) {
        errors.password = 'Mot de passe requis';
      } else if (value.length < 4) {
        errors.password = 'Au moins 4 caractères requis';
      }
    }
    
    return errors;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Live validation
    if (touched[name]) {
      const errors = validateField(name, value);
      setFieldErrors(prev => ({ ...prev, ...errors }));
      
      // Clear error if field is now valid
      if (!errors[name]) {
        setFieldErrors(prev => {
          const newErrors = { ...prev };
          delete newErrors[name];
          return newErrors;
        });
      }
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    
    const errors = validateField(name, value);
    setFieldErrors(prev => ({ ...prev, ...errors }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate all fields
    const usernameErrors = validateField('username', formData.username);
    const passwordErrors = validateField('password', formData.password);
    const allErrors = { ...usernameErrors, ...passwordErrors };
    
    setFieldErrors(allErrors);
    setTouched({ username: true, password: true });
    
    // If no errors, submit
    if (Object.keys(allErrors).length === 0) {
      onSubmit(formData);
    }
  };

  const isFormValid = Object.keys(fieldErrors).length === 0 && 
                     formData.username.trim() && 
                     formData.password;

  return (
    <Card className={cn("w-full max-w-md shadow-2xl border-0", className)}>
      <CardHeader className="space-y-4 pb-6">
        {/* Logo placeholder */}
        <div className="flex justify-center">
          <div className="p-4 rounded-full bg-primary/10 text-primary">
            <KeyRound className="h-8 w-8" />
          </div>
        </div>
        
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">
            M&A Intelligence
          </h1>
          <p className="text-muted-foreground">
            Connectez-vous pour accéder à la plateforme
          </p>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Error Alert */}
        {error && (
          <AlertWithIcon variant="destructive" className="animate-fade-in">
            {error}
          </AlertWithIcon>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Username Field */}
          <div className="space-y-2">
            <Label htmlFor="username" className="text-sm font-medium">
              Nom d'utilisateur
            </Label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="username"
                name="username"
                type="text"
                placeholder="Entrez votre nom d'utilisateur"
                value={formData.username}
                onChange={handleChange}
                onBlur={handleBlur}
                className={cn(
                  "pl-10 transition-all duration-200",
                  fieldErrors.username && touched.username
                    ? "border-destructive focus-visible:ring-destructive"
                    : "focus-visible:ring-primary",
                  !fieldErrors.username && touched.username && formData.username
                    ? "border-green-500 focus-visible:ring-green-500"
                    : ""
                )}
                disabled={isLoading}
              />
            </div>
            {fieldErrors.username && touched.username && (
              <p className="text-sm text-destructive animate-fade-in">
                {fieldErrors.username}
              </p>
            )}
          </div>

          {/* Password Field */}
          <div className="space-y-2">
            <Label htmlFor="password" className="text-sm font-medium">
              Mot de passe
            </Label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                id="password"
                name="password"
                type={showPassword ? "text" : "password"}
                placeholder="Entrez votre mot de passe"
                value={formData.password}
                onChange={handleChange}
                onBlur={handleBlur}
                className={cn(
                  "pl-10 pr-10 transition-all duration-200",
                  fieldErrors.password && touched.password
                    ? "border-destructive focus-visible:ring-destructive"
                    : "focus-visible:ring-primary",
                  !fieldErrors.password && touched.password && formData.password
                    ? "border-green-500 focus-visible:ring-green-500"
                    : ""
                )}
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                disabled={isLoading}
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            {fieldErrors.password && touched.password && (
              <p className="text-sm text-destructive animate-fade-in">
                {fieldErrors.password}
              </p>
            )}
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            className={cn(
              "w-full gap-2 transition-all duration-200",
              isFormValid && !isLoading && "hover:scale-[1.02]"
            )}
            disabled={!isFormValid || isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Connexion en cours...
              </>
            ) : (
              <>
                <LogIn className="h-4 w-4" />
                Se connecter
              </>
            )}
          </Button>
        </form>

        {/* Demo Credentials */}
        <div className="pt-4 border-t">
          <p className="text-xs text-muted-foreground text-center mb-3">
            Compte de démonstration
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <button
              type="button"
              onClick={() => {
                setFormData({ username: 'admin', password: 'secret' });
                setTouched({});
                setFieldErrors({});
              }}
              className="p-2 text-left border rounded hover:bg-muted/50 transition-colors"
              disabled={isLoading}
            >
              <div className="font-medium">Admin</div>
              <div className="text-muted-foreground">admin / secret</div>
            </button>
            <button
              type="button"
              onClick={() => {
                setFormData({ username: 'demo', password: 'demo123' });
                setTouched({});
                setFieldErrors({});
              }}
              className="p-2 text-left border rounded hover:bg-muted/50 transition-colors"
              disabled={isLoading}
            >
              <div className="font-medium">Demo</div>
              <div className="text-muted-foreground">demo / demo123</div>
            </button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}