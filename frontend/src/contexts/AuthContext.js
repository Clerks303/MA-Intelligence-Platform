import React, { createContext, useState, useContext, useEffect } from 'react';
import api from '../services/api';
import tokenStorage, { migrateLegacyStorage } from '../utils/tokenStorage';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Migrate legacy storage and check if user is logged in
    migrateLegacyStorage();
    
    const token = tokenStorage.getToken();
    const savedUser = tokenStorage.getUserData();
    
    console.log('🔍 Auth init - Token:', !!token, 'User:', !!savedUser);
    
    if (token && savedUser) {
      setUser(savedUser);
    }
    
    setLoading(false);
  }, []);

  const handleApiError = (error) => {
    console.error('🚨 API Error:', error.response?.status, error.response?.data);
    
    if (error.response?.status === 401) {
      // Token invalide ou expiré - déconnexion automatique
      logout();
      throw new Error('Session expirée. Veuillez vous reconnecter.');
    } else if (error.response?.status === 403) {
      // Accès refusé
      throw new Error('Accès non autorisé.');
    } else if (error.response?.status >= 500) {
      // Erreur serveur
      throw new Error('Erreur serveur. Veuillez réessayer plus tard.');
    } else if (error.response?.status === 422) {
      // Erreur de validation
      const detail = error.response?.data?.detail;
      if (typeof detail === 'string') {
        throw new Error(detail);
      } else if (Array.isArray(detail)) {
        throw new Error(`Erreur de validation: ${detail[0]?.msg || 'Données invalides'}`);
      } else {
        throw new Error('Identifiants incorrects.');
      }
    } else if (!error.response) {
      // Erreur réseau
      throw new Error('Impossible de contacter le serveur. Vérifiez votre connexion.');
    } else {
      // Autres erreurs
      const message = error.response?.data?.detail || error.response?.data?.message || error.message;
      throw new Error(message || 'Une erreur est survenue.');
    }
  };

  const login = async (username, password) => {
    try {
      console.log('🚀 Login attempt for:', username);
      
      // Correction : URL complète vers l'endpoint
      const response = await api.post('/auth/login', { 
        username: username.trim(), 
        password: password 
      });
      
      console.log('✅ Login response:', response.data);
      
      const { access_token, token_type } = response.data;
      
      if (!access_token) {
        console.error('❌ No access token in response');
        throw new Error('Réponse invalide du serveur.');
      }      
      
      // Save token with secure storage
      const tokenSaved = tokenStorage.setToken(access_token, 30); // 30 minutes expiry
      if (!tokenSaved) {
        throw new Error('Erreur lors de la sauvegarde du token.');
      }
      
      // Save user info
      const userInfo = { 
        username: username.trim(),
        authenticated_at: new Date().toISOString()
      };
      const userSaved = tokenStorage.setUserData(userInfo);
      if (!userSaved) {
        throw new Error('Erreur lors de la sauvegarde des données utilisateur.');
      }
      
      setUser(userInfo);
      
      console.log('✅ Login successful, user set:', userInfo);
      
      return response.data;
    } catch (error) {
      console.error('❌ Login failed:', error);
      handleApiError(error);
    }
  };

  const logout = () => {
    console.log('🚪 Logout initiated');
    tokenStorage.clearAuth();
    setUser(null);
  };

  const value = {
    user,
    login,
    logout,
    loading,
    isAuthenticated: !!user
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};