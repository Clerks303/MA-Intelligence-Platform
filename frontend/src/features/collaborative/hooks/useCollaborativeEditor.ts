/**
 * Hook principal pour édition collaborative - M&A Intelligence Platform
 * Sprint 5 - Intégration Y.js + Tiptap + WebSocket avec fallback polling
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  EditorConfig,
  EditorState,
  ConnectionState,
  CollaboratorInfo,
  PresenceInfo,
  Comment,
  CollaborationError,
  UseCollaborativeEditorReturn,
  WebSocketMessage,
  COLLABORATION_COLORS,
  DEFAULT_EDITOR_CONFIG,
  DEFAULT_CONNECTION_CONFIG
} from '../types';
import { CollaborativeWebSocketService } from '../services/websocketService';

// Simuler Y.js et Tiptap pour la démo (en attendant l'installation des dépendances)
interface MockYDoc {
  getText(name: string): MockYText;
  on(event: string, callback: Function): void;
  off(event: string, callback: Function): void;
}

interface MockYText {
  toString(): string;
  insert(index: number, content: string): void;
  delete(index: number, length: number): void;
  on(event: string, callback: Function): void;
}

interface MockEditor {
  getHTML(): string;
  setHTML(content: string): void;
  commands: {
    setContent: (content: string) => void;
    focus: () => void;
  };
  on(event: string, callback: Function): void;
  off(event: string, callback: Function): void;
  destroy(): void;
}

// Mock implementations
class MockYDocImpl implements MockYDoc {
  private content = '';
  private listeners = new Map<string, Function[]>();
  
  getText(name: string): MockYText {
    return new MockYTextImpl(this.content, (newContent: string) => {
      this.content = newContent;
      this.emit('update', new Uint8Array([]));
    });
  }
  
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }
  
  off(event: string, callback: Function): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }
  
  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
}

class MockYTextImpl implements MockYText {
  private listeners = new Map<string, Function[]>();
  
  constructor(private content: string, private onChange: (content: string) => void) {}
  
  toString(): string {
    return this.content;
  }
  
  insert(index: number, content: string): void {
    this.content = this.content.slice(0, index) + content + this.content.slice(index);
    this.onChange(this.content);
    this.emit('delta', { retain: index, insert: content });
  }
  
  delete(index: number, length: number): void {
    this.content = this.content.slice(0, index) + this.content.slice(index + length);
    this.onChange(this.content);
    this.emit('delta', { retain: index, delete: length });
  }
  
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }
  
  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
}

class MockEditorImpl implements MockEditor {
  private content = '';
  private listeners = new Map<string, Function[]>();
  
  getHTML(): string {
    return this.content;
  }
  
  setHTML(content: string): void {
    this.content = content;
    this.emit('update', { editor: this });
  }
  
  commands = {
    setContent: (content: string) => {
      this.setHTML(content);
    },
    focus: () => {
      console.log('Editor focused');
    }
  };
  
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }
  
  off(event: string, callback: Function): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }
  
  destroy(): void {
    this.listeners.clear();
  }
  
  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
}

export const useCollaborativeEditor = (
  config: EditorConfig
): UseCollaborativeEditorReturn => {
  // References
  const editorRef = useRef<MockEditor | null>(null);
  const ydocRef = useRef<MockYDoc | null>(null);
  const wsServiceRef = useRef<CollaborativeWebSocketService | null>(null);
  const ytextRef = useRef<MockYText | null>(null);
  
  // State
  const [editorState, setEditorState] = useState<EditorState>({
    isConnected: false,
    isCollaborative: false,
    hasUnsavedChanges: false,
    lastSaved: null,
    collaboratorCount: 0,
    currentUser: null
  });
  
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'disconnected',
    websocketAvailable: typeof WebSocket !== 'undefined',
    pollingFallback: false,
    lastConnected: null,
    reconnectAttempts: 0
  });
  
  const [collaborators, setCollaborators] = useState<CollaboratorInfo[]>([]);
  const [comments, setComments] = useState<Comment[]>([]);
  const [presence, setPresence] = useState<PresenceInfo>({
    collaborators: [],
    totalCount: 0,
    activeCount: 0
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<CollaborationError | null>(null);
  
  // Configuration combinée
  const fullConfig = useMemo(() => ({
    ...DEFAULT_EDITOR_CONFIG,
    ...config
  }), [config]);
  
  // Current user info
  const currentUser: CollaboratorInfo = useMemo(() => ({
    userId: fullConfig.userId,
    username: fullConfig.username,
    avatar: fullConfig.avatar,
    color: COLLABORATION_COLORS[0], // TODO: Assign unique color
    isActive: true,
    lastSeen: new Date().toISOString()
  }), [fullConfig]);
  
  // Initialize editor and collaboration
  useEffect(() => {
    const initializeEditor = async () => {
      setIsLoading(true);
      
      try {
        // Initialize Y.js document
        ydocRef.current = new MockYDocImpl();
        ytextRef.current = ydocRef.current.getText('default');
        
        // Initialize Tiptap editor  
        editorRef.current = new MockEditorImpl();
        
        // Setup collaboration bindings
        setupCollaborationBindings();
        
        // Initialize WebSocket service
        const wsConfig = {
          websocketUrl: fullConfig.websocketUrl,
          pollingUrl: fullConfig.fallbackPollingUrl || '/api/v1/collaborative/document/polling',
          authToken: 'mock-token', // TODO: Get real token
          ...DEFAULT_CONNECTION_CONFIG
        };
        
        wsServiceRef.current = new CollaborativeWebSocketService(wsConfig);
        
        // Setup WebSocket handlers
        setupWebSocketHandlers();
        
        setEditorState(prev => ({
          ...prev,
          currentUser,
          isCollaborative: true
        }));
        
      } catch (err) {
        console.error('Failed to initialize editor:', err);
        setError({
          code: 'INIT_ERROR',
          message: 'Failed to initialize collaborative editor',
          type: 'connection',
          recoverable: false,
          timestamp: Date.now()
        });
      } finally {
        setIsLoading(false);
      }
    };
    
    initializeEditor();
    
    // Cleanup
    return () => {
      cleanup();
    };
  }, [fullConfig, currentUser]);
  
  // Setup collaboration bindings between Y.js and Tiptap
  const setupCollaborationBindings = useCallback(() => {
    if (!ydocRef.current || !ytextRef.current || !editorRef.current) return;
    
    const ytext = ytextRef.current;
    const editor = editorRef.current;
    
    // Bind Y.js changes to editor
    const onYTextChange = () => {
      const content = ytext.toString();
      editor.commands.setContent(content);
      
      setEditorState(prev => ({
        ...prev,
        hasUnsavedChanges: true
      }));
    };
    
    ytext.on('delta', onYTextChange);
    
    // Bind editor changes to Y.js
    const onEditorUpdate = () => {
      const content = editor.getHTML();
      // In real implementation, this would properly sync with Y.js
      console.log('Editor updated:', content);
    };
    
    editor.on('update', onEditorUpdate);
    
  }, []);
  
  // Setup WebSocket message handlers
  const setupWebSocketHandlers = useCallback(() => {
    if (!wsServiceRef.current) return;
    
    const wsService = wsServiceRef.current;
    
    // Handle connection state changes
    wsService.onStateChange((state) => {
      setConnectionState(state);
      setEditorState(prev => ({
        ...prev,
        isConnected: state.status === 'connected'
      }));
    });
    
    // Handle Y.js updates
    wsService.onMessage('update', (message: WebSocketMessage) => {
      if (ydocRef.current && message.payload instanceof Uint8Array) {
        // In real implementation, apply Y.js update
        console.log('Received Y.js update:', message.payload);
      }
    });
    
    // Handle awareness (presence) updates
    wsService.onMessage('awareness', (message: WebSocketMessage) => {
      console.log('Received awareness update:', message.payload);
      updatePresence(message.payload);
    });
    
    // Handle sync messages
    wsService.onMessage('sync', (message: WebSocketMessage) => {
      console.log('Received sync message:', message.payload);
    });
    
    // Handle errors
    wsService.onMessage('error', (message: WebSocketMessage) => {
      setError({
        code: 'WEBSOCKET_ERROR',
        message: message.payload.message || 'WebSocket error',
        type: 'connection',
        recoverable: true,
        timestamp: Date.now()
      });
    });
    
  }, []);
  
  // Update presence information
  const updatePresence = useCallback((awarenessData: any) => {
    // Mock presence update
    const mockCollaborators: CollaboratorInfo[] = [
      {
        userId: 'user2',
        username: 'John Doe',
        color: COLLABORATION_COLORS[1],
        isActive: true,
        lastSeen: new Date().toISOString()
      },
      {
        userId: 'user3',
        username: 'Jane Smith',
        color: COLLABORATION_COLORS[2],
        isActive: true,
        lastSeen: new Date().toISOString()
      }
    ];
    
    setCollaborators(mockCollaborators);
    setPresence({
      collaborators: mockCollaborators,
      totalCount: mockCollaborators.length + 1, // +1 for current user
      activeCount: mockCollaborators.filter(c => c.isActive).length + 1
    });
    
    setEditorState(prev => ({
      ...prev,
      collaboratorCount: mockCollaborators.length
    }));
  }, []);
  
  // Public API methods
  const connect = useCallback(async (): Promise<void> => {
    if (!wsServiceRef.current) return;
    
    setIsConnecting(true);
    setError(null);
    
    try {
      await wsServiceRef.current.connect();
    } catch (err) {
      setError({
        code: 'CONNECTION_ERROR',
        message: 'Failed to connect to collaboration service',
        type: 'connection',
        recoverable: true,
        timestamp: Date.now()
      });
    } finally {
      setIsConnecting(false);
    }
  }, []);
  
  const disconnect = useCallback((): void => {
    if (wsServiceRef.current) {
      wsServiceRef.current.disconnect();
    }
  }, []);
  
  const saveDocument = useCallback(async (): Promise<void> => {
    if (!editorRef.current) return;
    
    setIsSaving(true);
    
    try {
      const content = editorRef.current.getHTML();
      
      // Mock save API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setEditorState(prev => ({
        ...prev,
        hasUnsavedChanges: false,
        lastSaved: new Date().toISOString()
      }));
      
    } catch (err) {
      setError({
        code: 'SAVE_ERROR',
        message: 'Failed to save document',
        type: 'sync',
        recoverable: true,
        timestamp: Date.now()
      });
    } finally {
      setIsSaving(false);
    }
  }, []);
  
  const addComment = useCallback(async (position: number, content: string): Promise<void> => {
    const comment: Comment = {
      id: `comment-${Date.now()}`,
      documentId: fullConfig.documentId,
      position,
      content,
      author: currentUser,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      isResolved: false
    };
    
    setComments(prev => [...prev, comment]);
    
    // Send comment via WebSocket
    if (wsServiceRef.current) {
      wsServiceRef.current.send({
        type: 'sync',
        payload: {
          action: 'add_comment',
          comment
        },
        timestamp: Date.now()
      });
    }
  }, [fullConfig.documentId, currentUser]);
  
  const resolveComment = useCallback(async (commentId: string): Promise<void> => {
    setComments(prev => 
      prev.map(comment => 
        comment.id === commentId 
          ? { ...comment, isResolved: true, updatedAt: new Date().toISOString() }
          : comment
      )
    );
    
    // Send resolution via WebSocket
    if (wsServiceRef.current) {
      wsServiceRef.current.send({
        type: 'sync',
        payload: {
          action: 'resolve_comment',
          commentId
        },
        timestamp: Date.now()
      });
    }
  }, []);
  
  // Auto-save functionality
  useEffect(() => {
    if (!fullConfig.autoSave || !editorState.hasUnsavedChanges) return;
    
    const autoSaveTimer = setTimeout(() => {
      saveDocument();
    }, fullConfig.autoSaveInterval || 5000);
    
    return () => clearTimeout(autoSaveTimer);
  }, [editorState.hasUnsavedChanges, fullConfig.autoSave, fullConfig.autoSaveInterval, saveDocument]);
  
  // Cleanup function
  const cleanup = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.destroy();
    }
    
    if (wsServiceRef.current) {
      wsServiceRef.current.disconnect();
    }
    
    // Reset state
    setEditorState({
      isConnected: false,
      isCollaborative: false,
      hasUnsavedChanges: false,
      lastSaved: null,
      collaboratorCount: 0,
      currentUser: null
    });
  }, []);
  
  return {
    // Editor state
    editorState,
    connectionState,
    
    // Collaboration data
    collaborators,
    comments,
    presence,
    
    // Actions
    connect,
    disconnect,
    saveDocument,
    addComment,
    resolveComment,
    
    // Editor ref
    editorRef,
    
    // Loading states
    isLoading,
    isSaving,
    isConnecting,
    
    // Errors
    error
  };
};

export default useCollaborativeEditor;