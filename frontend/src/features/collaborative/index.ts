/**
 * Index pour les fonctionnalités collaboratives - M&A Intelligence Platform
 * Sprint 5 - Exports principaux pour l'édition collaborative
 */

// Composants principaux
export { default as CollaborativeEditor } from './components/CollaborativeEditor';
export { default as CollaborativeTestPage } from './components/CollaborativeTestPage';

// Hooks
export { useCollaborativeEditor } from './hooks/useCollaborativeEditor';

// Services
export { CollaborativeWebSocketService, createWebSocketService } from './services/websocketService';

// Types
export type {
  EditorConfig,
  EditorState,
  ConnectionState,
  CollaboratorInfo,
  PresenceInfo,
  Comment,
  CollaborationError,
  UseCollaborativeEditorReturn,
  WebSocketMessage,
  YjsDocument,
  YjsUpdate,
  YjsAwareness,
  CollaborativeEvent,
  Operation,
  PollingResponse
} from './types';

// Constantes
export {
  COLLABORATION_COLORS,
  WEBSOCKET_READY_STATES,
  MESSAGE_TYPES,
  DEFAULT_EDITOR_CONFIG,
  DEFAULT_POLLING_CONFIG,
  DEFAULT_CONNECTION_CONFIG
} from './types';

// Fonctions utilitaires pour l'intégration
export const collaborativeFeatures = {
  // Configuration par défaut pour Y.js
  getDefaultYjsConfig: (documentId: string, userId: string) => ({
    documentId,
    userId,
    websocketUrl: `ws://localhost:8000/api/v1/collaborative/ws/yjs/${documentId}`,
    fallbackPollingUrl: '/api/v1/collaborative/document/polling'
  }),
  
  // Couleurs pour les collaborateurs
  getCollaboratorColor: (index: number) => 
    COLLABORATION_COLORS[index % COLLABORATION_COLORS.length],
  
  // Validation des configurations
  validateConfig: (config: Partial<EditorConfig>) => {
    const required = ['documentId', 'userId', 'username', 'websocketUrl'];
    const missing = required.filter(field => !config[field as keyof EditorConfig]);
    
    if (missing.length > 0) {
      throw new Error(`Configuration manquante: ${missing.join(', ')}`);
    }
    
    return true;
  },
  
  // Formatage des messages d'erreur
  formatCollaborationError: (error: CollaborationError) => {
    const prefix = error.recoverable ? '⚠️' : '❌';
    return `${prefix} ${error.code}: ${error.message}`;
  }
};

export default {
  CollaborativeEditor,
  CollaborativeTestPage,
  useCollaborativeEditor,
  CollaborativeWebSocketService,
  collaborativeFeatures
};