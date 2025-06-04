/**
 * Hooks Authentification Sécurisée - M&A Intelligence Platform
 * Sprint 4 - Hooks pour auth, MFA et sessions
 */

import { useState, useCallback, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  User,
  SecurityLevel,
  UserSession,
  MFAStatus,
  ChangePasswordRequest,
  MFAVerifyRequest,
  LoginResponse
} from '../types/api';
import { securityService } from '../services/securityService';

// === AUTHENTIFICATION DE BASE ===

export const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(
    () => !!localStorage.getItem('access_token')
  );
  const [mfaRequired, setMfaRequired] = useState(false);
  const [mfaToken, setMfaToken] = useState<string | null>(null);

  const queryClient = useQueryClient();

  // Profil utilisateur
  const { 
    data: currentUser, 
    isLoading: isLoadingProfile,
    error: profileError,
    refetch: refetchProfile
  } = useQuery({
    queryKey: ['auth', 'profile'],
    queryFn: securityService.auth.getProfile,
    enabled: isAuthenticated && !mfaRequired,
    retry: false,
  });

  // Mutation login
  const loginMutation = useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) =>
      securityService.auth.loginJSON(username, password),
    onSuccess: (data: LoginResponse) => {
      if (data.requires_mfa && data.mfa_token) {
        setMfaRequired(true);
        setMfaToken(data.mfa_token);
      } else {
        // Login réussi sans MFA
        setIsAuthenticated(true);
        setMfaRequired(false);
        queryClient.invalidateQueries({ queryKey: ['auth'] });
      }
    },
  });

  // Mutation login standard OAuth2
  const loginOAuth2Mutation = useMutation({
    mutationFn: ({ username, password }: { username: string; password: string }) =>
      securityService.auth.login(username, password),
    onSuccess: (data: LoginResponse) => {
      localStorage.setItem('access_token', data.access_token);
      setIsAuthenticated(true);
      setMfaRequired(false);
      queryClient.invalidateQueries({ queryKey: ['auth'] });
    },
  });

  // Mutation logout
  const logoutMutation = useMutation({
    mutationFn: securityService.auth.logout,
    onSuccess: () => {
      setIsAuthenticated(false);
      setMfaRequired(false);
      setMfaToken(null);
      queryClient.clear();
    },
    onError: () => {
      // Même en cas d'erreur, déconnecter localement
      setIsAuthenticated(false);
      setMfaRequired(false);
      setMfaToken(null);
      localStorage.removeItem('access_token');
      queryClient.clear();
    },
  });

  // Changement de mot de passe
  const changePasswordMutation = useMutation({
    mutationFn: (data: ChangePasswordRequest) =>
      securityService.auth.changePassword(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auth'] });
    },
  });

  // Actions
  const login = useCallback((username: string, password: string) => {
    loginMutation.mutate({ username, password });
  }, [loginMutation]);

  const loginOAuth2 = useCallback((username: string, password: string) => {
    loginOAuth2Mutation.mutate({ username, password });
  }, [loginOAuth2Mutation]);

  const logout = useCallback(() => {
    logoutMutation.mutate();
  }, [logoutMutation]);

  const changePassword = useCallback((data: ChangePasswordRequest) => {
    changePasswordMutation.mutate(data);
  }, [changePasswordMutation]);

  // Vérifier le token au chargement
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token && !isAuthenticated) {
      setIsAuthenticated(true);
    }
  }, [isAuthenticated]);

  return {
    // État
    isAuthenticated,
    currentUser,
    mfaRequired,
    mfaToken,
    
    // Chargement
    isLoading: isLoadingProfile,
    isLoggingIn: loginMutation.isPending || loginOAuth2Mutation.isPending,
    isLoggingOut: logoutMutation.isPending,
    isChangingPassword: changePasswordMutation.isPending,
    
    // Erreurs
    loginError: loginMutation.error || loginOAuth2Mutation.error,
    profileError,
    changePasswordError: changePasswordMutation.error,
    
    // Actions
    login,
    loginOAuth2,
    logout,
    changePassword,
    refetchProfile,
    
    // Données mutations
    loginData: loginMutation.data,
    changePasswordData: changePasswordMutation.data,
  };
};

// === MFA (MULTI-FACTOR AUTHENTICATION) ===

export const useMFA = () => {
  const queryClient = useQueryClient();

  // Statut MFA
  const { 
    data: mfaStatus, 
    isLoading: isLoadingStatus,
    refetch: refetchStatus 
  } = useQuery({
    queryKey: ['auth', 'mfa', 'status'],
    queryFn: securityService.mfa.getStatus,
    retry: false,
  });

  // Setup MFA
  const setupMutation = useMutation({
    mutationFn: securityService.mfa.setup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfa'] });
    },
  });

  // Vérifier MFA
  const verifyMutation = useMutation({
    mutationFn: (data: MFAVerifyRequest) => securityService.mfa.verify(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfa'] });
      queryClient.invalidateQueries({ queryKey: ['auth', 'profile'] });
    },
  });

  // Vérifier code lors du login
  const verifyLoginMutation = useMutation({
    mutationFn: ({ mfaToken, code }: { mfaToken: string; code: string }) =>
      securityService.mfa.verifyLogin(mfaToken, code),
    onSuccess: (data: LoginResponse) => {
      localStorage.setItem('access_token', data.access_token);
      queryClient.invalidateQueries({ queryKey: ['auth'] });
    },
  });

  // Désactiver MFA
  const disableMutation = useMutation({
    mutationFn: (password: string) => securityService.mfa.disable(password),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfa'] });
      queryClient.invalidateQueries({ queryKey: ['auth', 'profile'] });
    },
  });

  // Générer codes de backup
  const generateBackupCodesMutation = useMutation({
    mutationFn: securityService.mfa.generateBackupCodes,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfa'] });
    },
  });

  // Actions
  const setupMFA = useCallback(() => {
    setupMutation.mutate();
  }, [setupMutation]);

  const verifyMFA = useCallback((data: MFAVerifyRequest) => {
    verifyMutation.mutate(data);
  }, [verifyMutation]);

  const verifyMFALogin = useCallback((mfaToken: string, code: string) => {
    verifyLoginMutation.mutate({ mfaToken, code });
  }, [verifyLoginMutation]);

  const disableMFA = useCallback((password: string) => {
    disableMutation.mutate(password);
  }, [disableMutation]);

  const generateBackupCodes = useCallback(() => {
    generateBackupCodesMutation.mutate();
  }, [generateBackupCodesMutation]);

  return {
    // État
    mfaStatus,
    isEnabled: (mfaStatus as MFAStatus)?.enabled || false,
    isSetupComplete: (mfaStatus as MFAStatus)?.setup_complete || false,
    devices: (mfaStatus as MFAStatus)?.devices || [],
    backupCodesRemaining: (mfaStatus as MFAStatus)?.backup_codes_remaining || 0,
    
    // Chargement
    isLoading: isLoadingStatus,
    isSettingUp: setupMutation.isPending,
    isVerifying: verifyMutation.isPending,
    isVerifyingLogin: verifyLoginMutation.isPending,
    isDisabling: disableMutation.isPending,
    isGeneratingBackupCodes: generateBackupCodesMutation.isPending,
    
    // Erreurs
    setupError: setupMutation.error,
    verifyError: verifyMutation.error,
    verifyLoginError: verifyLoginMutation.error,
    disableError: disableMutation.error,
    generateBackupCodesError: generateBackupCodesMutation.error,
    
    // Actions
    setupMFA,
    verifyMFA,
    verifyMFALogin,
    disableMFA,
    generateBackupCodes,
    refetchStatus,
    
    // Données
    setupData: setupMutation.data,
    verifyData: verifyMutation.data,
    verifyLoginData: verifyLoginMutation.data,
    backupCodesData: generateBackupCodesMutation.data,
  };
};

// === SESSIONS ===

export const useSessions = (userId?: string) => {
  const queryClient = useQueryClient();

  // Sessions actives
  const { 
    data: sessions = [], 
    isLoading,
    refetch 
  } = useQuery({
    queryKey: ['security', 'sessions', userId],
    queryFn: () => securityService.sessions.getActiveSessions(
      userId ? { user_id: userId } : undefined
    ),
  });

  // Session courante
  const { 
    data: currentSession,
    isLoading: isLoadingCurrent 
  } = useQuery({
    queryKey: ['security', 'sessions', 'current'],
    queryFn: securityService.sessions.getCurrentSession,
    retry: false,
  });

  // Révoquer session
  const revokeSessionMutation = useMutation({
    mutationFn: (sessionId: string) => securityService.sessions.revokeSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'sessions'] });
    },
  });

  // Prolonger session
  const extendSessionMutation = useMutation({
    mutationFn: (sessionId: string) => securityService.sessions.extendSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'sessions'] });
    },
  });

  // Actions
  const revokeSession = useCallback((sessionId: string) => {
    revokeSessionMutation.mutate(sessionId);
  }, [revokeSessionMutation]);

  const extendSession = useCallback((sessionId: string) => {
    extendSessionMutation.mutate(sessionId);
  }, [extendSessionMutation]);

  // Calculer les métriques de sessions
  const sessionMetrics = {
    total: sessions.length,
    active: sessions.filter((s: UserSession) => s.is_current).length,
    suspicious: sessions.filter((s: UserSession) => s.is_suspicious).length,
    bySecurityLevel: sessions.reduce((acc: Record<SecurityLevel, number>, session: UserSession) => {
      acc[session.security_level] = (acc[session.security_level] || 0) + 1;
      return acc;
    }, {} as Record<SecurityLevel, number>),
    byLocation: sessions.reduce((acc: Record<string, number>, session: UserSession) => {
      const location = session.location ? 
        `${session.location.city}, ${session.location.country}` : 
        'Unknown';
      acc[location] = (acc[location] || 0) + 1;
      return acc;
    }, {} as Record<string, number>),
  };

  return {
    // Données
    sessions,
    currentSession,
    sessionMetrics,
    
    // Chargement
    isLoading: isLoading || isLoadingCurrent,
    isRevokingSession: revokeSessionMutation.isPending,
    isExtendingSession: extendSessionMutation.isPending,
    
    // Erreurs
    revokeError: revokeSessionMutation.error,
    extendError: extendSessionMutation.error,
    
    // Actions
    revokeSession,
    extendSession,
    refetch,
    
    // Données mutations
    revokeData: revokeSessionMutation.data,
    extendData: extendSessionMutation.data,
  };
};

// === HOOK COMBINÉ POUR L'AUTHENTIFICATION COMPLÈTE ===

export const useSecurityAuth = () => {
  const auth = useAuth();
  const mfa = useMFA();
  const sessions = useSessions();

  // État de sécurité global
  const securityLevel: SecurityLevel = 
    (sessions.currentSession as UserSession)?.security_level || 
    ((auth.currentUser as User)?.is_superuser ? 'HIGH' : 'MEDIUM');

  const isSecureSession = securityLevel === 'HIGH' || securityLevel === 'CRITICAL';
  
  const requiresAdditionalAuth = (action: 'sensitive' | 'admin' | 'critical') => {
    const thresholds = {
      sensitive: 'MEDIUM',
      admin: 'HIGH', 
      critical: 'CRITICAL',
    };
    
    const required = thresholds[action];
    const levels = { LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
    
    return levels[securityLevel] < levels[required as SecurityLevel];
  };

  return {
    // Auth de base
    ...auth,
    
    // MFA
    mfa: {
      ...mfa,
      isRequired: auth.mfaRequired,
      token: auth.mfaToken,
    },
    
    // Sessions
    sessions,
    
    // Sécurité
    securityLevel,
    isSecureSession,
    requiresAdditionalAuth,
    
    // État global
    isFullyAuthenticated: auth.isAuthenticated && !auth.mfaRequired && !!auth.currentUser,
    isSecurityLoading: auth.isLoading || mfa.isLoading || sessions.isLoading,
  };
};

export default {
  useAuth,
  useMFA,
  useSessions,
  useSecurityAuth,
};