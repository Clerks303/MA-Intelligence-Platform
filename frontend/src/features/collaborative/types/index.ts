/**
 * Types pour édition collaborative - M&A Intelligence Platform
 * Sprint 5 - Y.js + Tiptap + WebSocket
 */

// === Y.JS TYPES ===

export interface YjsDocument {
  documentId: string;
  content: string;
  version: number;
  lastUpdated: string;
  activeCollaborators: number;
}

export interface YjsUpdate {
  update: Uint8Array;
  origin?: string;
  timestamp: number;
}

export interface YjsAwareness {
  clientId: string;
  user: CollaboratorInfo;
  cursor?: CursorPosition;
  selection?: SelectionRange;
  timestamp: number;
}

export interface CursorPosition {
  anchor: number;
  head: number;
}

export interface SelectionRange {
  from: number;
  to: number;
}

// === COLLABORATION TYPES ===

export interface CollaboratorInfo {
  userId: string;
  username: string;
  avatar?: string;
  color: string;
  isActive: boolean;
  lastSeen: string;
}

export interface PresenceInfo {
  collaborators: CollaboratorInfo[];
  totalCount: number;
  activeCount: number;
}

export interface CollaborationSession {
  sessionId: string;
  documentId: string;
  createdAt: string;
  lastActivity: string;
  collaborators: CollaboratorInfo[];
  isActive: boolean;
}

// === WEBSOCKET TYPES ===

export interface WebSocketMessage {
  type: 'sync' | 'update' | 'awareness' | 'auth' | 'error';
  payload: any;
  timestamp: number;
  clientId?: string;
}

export interface SyncMessage {
  type: 'sync';
  payload: {
    step: 1 | 2;
    update?: Uint8Array;
    stateVector?: Uint8Array;
  };
}

export interface UpdateMessage {
  type: 'update';
  payload: {
    update: Uint8Array;
    origin?: string;
  };
}

export interface AwarenessMessage {
  type: 'awareness';
  payload: {
    awareness: Uint8Array;
    added?: number[];
    updated?: number[];
    removed?: number[];
  };
}

// === EDITOR TYPES ===

export interface EditorState {
  isConnected: boolean;
  isCollaborative: boolean;
  hasUnsavedChanges: boolean;
  lastSaved: string | null;
  collaboratorCount: number;
  currentUser: CollaboratorInfo | null;
}

export interface EditorConfig {
  documentId: string;
  userId: string;
  username: string;
  avatar?: string;
  websocketUrl: string;
  fallbackPollingUrl?: string;
  readOnly?: boolean;
  autoSave?: boolean;
  autoSaveInterval?: number;
}

export interface Comment {
  id: string;
  documentId: string;
  position: number;
  content: string;
  author: CollaboratorInfo;
  createdAt: string;
  updatedAt: string;
  isResolved: boolean;
  replies?: Comment[];
  mentions?: string[];
}

export interface DocumentVersion {
  version: number;
  timestamp: string;
  author: CollaboratorInfo;
  description?: string;
  contentSnapshot?: string;
}

// === POLLING FALLBACK TYPES ===

export interface PollingConfig {
  enabled: boolean;
  interval: number;
  maxRetries: number;
  backoffFactor: number;
}

export interface PollingResponse {
  hasUpdates: boolean;
  currentVersion: number;
  content: string;
  operations: Operation[];
  comments: Comment[];
  presence: PresenceInfo;
  timestamp: string;
}

export interface Operation {
  id: string;
  type: 'insert' | 'delete' | 'retain' | 'format';
  position: number;
  content?: string;
  length?: number;
  attributes?: Record<string, any>;
  userId: string;
  timestamp: string;
  version: number;
}

// === CONNECTION TYPES ===

export interface ConnectionState {
  status: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';
  websocketAvailable: boolean;
  pollingFallback: boolean;
  lastConnected: string | null;
  reconnectAttempts: number;
  error?: string;
}

export interface ConnectionConfig {
  websocketUrl: string;
  pollingUrl: string;
  authToken: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
}

// === EVENT TYPES ===

export interface CollaborativeEvent {
  type: 'user-joined' | 'user-left' | 'content-changed' | 'comment-added' | 'document-saved' | 'connection-changed';
  payload: any;
  timestamp: number;
  source: 'local' | 'remote';
}

export interface UserJoinedEvent extends CollaborativeEvent {
  type: 'user-joined';
  payload: {
    user: CollaboratorInfo;
    sessionId: string;
  };
}

export interface UserLeftEvent extends CollaborativeEvent {
  type: 'user-left';
  payload: {
    userId: string;
    sessionId: string;
  };
}

export interface ContentChangedEvent extends CollaborativeEvent {
  type: 'content-changed';
  payload: {
    operation: Operation;
    author: CollaboratorInfo;
  };
}

// === ERROR TYPES ===

export interface CollaborationError {
  code: string;
  message: string;
  type: 'connection' | 'sync' | 'auth' | 'permission' | 'conflict';
  recoverable: boolean;
  timestamp: number;
  context?: Record<string, any>;
}

// === HOOKS RETURN TYPES ===

export interface UseCollaborativeEditorReturn {
  // Editor state
  editorState: EditorState;
  connectionState: ConnectionState;
  
  // Collaboration data
  collaborators: CollaboratorInfo[];
  comments: Comment[];
  presence: PresenceInfo;
  
  // Actions
  connect: () => Promise<void>;
  disconnect: () => void;
  saveDocument: () => Promise<void>;
  addComment: (position: number, content: string) => Promise<void>;
  resolveComment: (commentId: string) => Promise<void>;
  
  // Editor ref
  editorRef: React.RefObject<any>;
  
  // Loading states
  isLoading: boolean;
  isSaving: boolean;
  isConnecting: boolean;
  
  // Errors
  error: CollaborationError | null;
}

export interface UseWebSocketReturn {
  // Connection state
  isConnected: boolean;
  connectionState: ConnectionState;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  send: (message: WebSocketMessage) => void;
  
  // Events
  onMessage: (callback: (message: WebSocketMessage) => void) => void;
  onStateChange: (callback: (state: ConnectionState) => void) => void;
  
  // Statistics
  messagesSent: number;
  messagesReceived: number;
  lastPing: number | null;
}

export interface UsePresenceReturn {
  // Presence data
  collaborators: CollaboratorInfo[];
  totalCount: number;
  activeCount: number;
  
  // Current user
  currentUser: CollaboratorInfo | null;
  
  // Actions
  updatePresence: (presence: Partial<YjsAwareness>) => void;
  setUserColor: (color: string) => void;
  
  // Loading
  isLoading: boolean;
  error: string | null;
}

// === UTILITY TYPES ===

export type CollaborationMode = 'realtime' | 'polling' | 'offline';

export type EditorPermission = 'read' | 'write' | 'comment' | 'admin';

export type DocumentFormat = 'markdown' | 'html' | 'json' | 'plaintext';

export interface ExportOptions {
  format: DocumentFormat;
  includeComments: boolean;
  includeVersionHistory: boolean;
  dateRange?: {
    from: string;
    to: string;
  };
}

// === DEFAULT VALUES ===

export const DEFAULT_EDITOR_CONFIG: Partial<EditorConfig> = {
  readOnly: false,
  autoSave: true,
  autoSaveInterval: 5000, // 5 seconds
};

export const DEFAULT_POLLING_CONFIG: PollingConfig = {
  enabled: true,
  interval: 3000, // 3 seconds
  maxRetries: 5,
  backoffFactor: 1.5,
};

export const DEFAULT_CONNECTION_CONFIG: Partial<ConnectionConfig> = {
  reconnectInterval: 1000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000, // 30 seconds
};

// === CONSTANTS ===

export const COLLABORATION_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
  '#786FA6', '#F8B500', '#78E08F', '#EA5455', '#3742FA'
];

export const WEBSOCKET_READY_STATES = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
} as const;

export const MESSAGE_TYPES = {
  SYNC_STEP_1: 0,
  SYNC_STEP_2: 1,
  UPDATE: 2,
  AWARENESS: 3,
  QUERY_AWARENESS: 4,
  AUTH: 5,
} as const;

export default {
  // Type exports for convenience
  YjsDocument,
  CollaboratorInfo,
  PresenceInfo,
  CollaborationSession,
  WebSocketMessage,
  EditorState,
  EditorConfig,
  Comment,
  ConnectionState,
  CollaborativeEvent,
  CollaborationError,
  
  // Constants
  COLLABORATION_COLORS,
  WEBSOCKET_READY_STATES,
  MESSAGE_TYPES,
  
  // Defaults
  DEFAULT_EDITOR_CONFIG,
  DEFAULT_POLLING_CONFIG,
  DEFAULT_CONNECTION_CONFIG,
};