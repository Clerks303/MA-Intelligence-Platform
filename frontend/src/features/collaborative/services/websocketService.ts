/**
 * WebSocket Service pour édition collaborative - M&A Intelligence Platform
 * Sprint 5 - Gestion des connexions WebSocket avec fallback polling
 */

import {
  WebSocketMessage,
  ConnectionState,
  ConnectionConfig,
  CollaborationError,
  MESSAGE_TYPES,
  WEBSOCKET_READY_STATES
} from '../types';

export class CollaborativeWebSocketService {
  private ws: WebSocket | null = null;
  private config: ConnectionConfig;
  private connectionState: ConnectionState;
  private messageQueue: WebSocketMessage[] = [];
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private messageHandlers: Map<string, (message: WebSocketMessage) => void> = new Map();
  private stateChangeHandlers: ((state: ConnectionState) => void)[] = [];
  
  // Statistics
  private messagesSent = 0;
  private messagesReceived = 0;
  private lastPing: number | null = null;
  
  constructor(config: ConnectionConfig) {
    this.config = config;
    this.connectionState = {
      status: 'disconnected',
      websocketAvailable: typeof WebSocket !== 'undefined',
      pollingFallback: false,
      lastConnected: null,
      reconnectAttempts: 0,
      error: undefined
    };
  }
  
  // === PUBLIC API ===
  
  async connect(): Promise<void> {
    if (this.connectionState.status === 'connected' || this.connectionState.status === 'connecting') {
      return;
    }
    
    this.updateConnectionState({ status: 'connecting' });
    
    try {
      if (!this.connectionState.websocketAvailable) {
        throw new Error('WebSocket not available, using polling fallback');
      }
      
      await this.connectWebSocket();
      
    } catch (error) {
      console.warn('WebSocket connection failed, falling back to polling:', error);
      this.fallbackToPolling();
    }
  }
  
  disconnect(): void {
    this.cleanup();
    this.updateConnectionState({ 
      status: 'disconnected',
      pollingFallback: false,
      error: undefined
    });
  }
  
  send(message: WebSocketMessage): void {
    if (this.connectionState.status !== 'connected') {
      // Queue message for when connection is restored
      this.messageQueue.push(message);
      return;
    }
    
    if (this.connectionState.pollingFallback) {
      // Handle via polling
      this.sendViaPolling(message);
    } else {
      // Send via WebSocket
      this.sendViaWebSocket(message);
    }
  }
  
  onMessage(type: string, handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.set(type, handler);
  }
  
  onStateChange(handler: (state: ConnectionState) => void): void {
    this.stateChangeHandlers.push(handler);
  }
  
  getConnectionState(): ConnectionState {
    return { ...this.connectionState };
  }
  
  getStatistics() {
    return {
      messagesSent: this.messagesSent,
      messagesReceived: this.messagesReceived,
      lastPing: this.lastPing,
      queuedMessages: this.messageQueue.length
    };
  }
  
  // === WEBSOCKET CONNECTION ===
  
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = this.buildWebSocketUrl();
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          this.updateConnectionState({
            status: 'connected',
            lastConnected: new Date().toISOString(),
            reconnectAttempts: 0,
            error: undefined
          });
          
          this.startHeartbeat();
          this.processMessageQueue();
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          this.handleWebSocketMessage(event);
        };
        
        this.ws.onclose = (event) => {
          this.handleWebSocketClose(event);
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(new Error('WebSocket connection failed'));
        };
        
        // Timeout for connection
        setTimeout(() => {
          if (this.ws?.readyState !== WEBSOCKET_READY_STATES.OPEN) {
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000); // 10 second timeout
        
      } catch (error) {
        reject(error);
      }
    });
  }
  
  private buildWebSocketUrl(): string {
    const baseUrl = this.config.websocketUrl;
    const params = new URLSearchParams({
      token: this.config.authToken,
    });
    
    return `${baseUrl}?${params.toString()}`;
  }
  
  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      let message: WebSocketMessage;
      
      // Handle binary messages (Y.js protocol)
      if (event.data instanceof ArrayBuffer || event.data instanceof Uint8Array) {
        message = {
          type: 'update',
          payload: new Uint8Array(event.data),
          timestamp: Date.now()
        };
      } else {
        // Handle JSON messages
        message = JSON.parse(event.data);
      }
      
      this.messagesReceived++;
      this.distributeMessage(message);
      
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  }
  
  private handleWebSocketClose(event: CloseEvent): void {
    console.log('WebSocket closed:', event.code, event.reason);
    
    this.cleanup();
    
    if (event.code !== 1000) { // Not a normal closure
      this.attemptReconnect();
    } else {
      this.updateConnectionState({ status: 'disconnected' });
    }
  }
  
  private sendViaWebSocket(message: WebSocketMessage): void {
    if (!this.ws || this.ws.readyState !== WEBSOCKET_READY_STATES.OPEN) {
      this.messageQueue.push(message);
      return;
    }
    
    try {
      // Handle binary messages (Y.js updates)
      if (message.type === 'update' && message.payload instanceof Uint8Array) {
        const buffer = new ArrayBuffer(message.payload.length + 1);
        const view = new Uint8Array(buffer);
        view[0] = MESSAGE_TYPES.UPDATE; // Y.js update type
        view.set(message.payload, 1);
        this.ws.send(buffer);
      } else {
        // Handle JSON messages
        this.ws.send(JSON.stringify(message));
      }
      
      this.messagesSent++;
      
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      this.messageQueue.push(message);
    }
  }
  
  // === POLLING FALLBACK ===
  
  private fallbackToPolling(): void {
    this.updateConnectionState({
      status: 'connected',
      pollingFallback: true,
      websocketAvailable: false
    });
    
    this.startPolling();
  }
  
  private startPolling(): void {
    // Implement polling logic
    const poll = async () => {
      try {
        const response = await fetch(this.config.pollingUrl, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.config.authToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            lastVersion: 0 // TODO: Track version
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          if (data.hasUpdates) {
            // Process updates
            const message: WebSocketMessage = {
              type: 'sync',
              payload: data,
              timestamp: Date.now()
            };
            
            this.distributeMessage(message);
          }
        }
        
      } catch (error) {
        console.error('Polling error:', error);
      }
      
      // Schedule next poll
      if (this.connectionState.pollingFallback) {
        setTimeout(poll, 3000); // Poll every 3 seconds
      }
    };
    
    poll();
  }
  
  private sendViaPolling(message: WebSocketMessage): void {
    // Queue for batch sending via polling
    this.messageQueue.push(message);
    
    // Send queued messages
    this.sendQueuedMessagesViaPolling();
  }
  
  private async sendQueuedMessagesViaPolling(): Promise<void> {
    if (this.messageQueue.length === 0) return;
    
    try {
      const response = await fetch(this.config.pollingUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.config.authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          messages: this.messageQueue
        })
      });
      
      if (response.ok) {
        this.messagesSent += this.messageQueue.length;
        this.messageQueue = [];
      }
      
    } catch (error) {
      console.error('Error sending messages via polling:', error);
    }
  }
  
  // === RECONNECTION ===
  
  private attemptReconnect(): void {
    if (this.connectionState.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.updateConnectionState({
        status: 'error',
        error: 'Max reconnect attempts reached'
      });
      return;
    }
    
    this.updateConnectionState({ status: 'reconnecting' });
    
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.connectionState.reconnectAttempts),
      30000 // Max 30 seconds
    );
    
    this.reconnectTimeout = setTimeout(() => {
      this.connectionState.reconnectAttempts++;
      this.connect();
    }, delay);
  }
  
  // === HEARTBEAT ===
  
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WEBSOCKET_READY_STATES.OPEN) {
        const ping = {
          type: 'ping',
          payload: {},
          timestamp: Date.now()
        };
        
        this.sendViaWebSocket(ping);
        this.lastPing = Date.now();
      }
    }, this.config.heartbeatInterval);
  }
  
  // === UTILITY METHODS ===
  
  private distributeMessage(message: WebSocketMessage): void {
    // Handle specific message types
    if (message.type === 'pong' && this.lastPing) {
      const latency = Date.now() - this.lastPing;
      console.log(`WebSocket latency: ${latency}ms`);
    }
    
    // Distribute to registered handlers
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message);
    }
    
    // Distribute to wildcard handlers
    const wildcardHandler = this.messageHandlers.get('*');
    if (wildcardHandler) {
      wildcardHandler(message);
    }
  }
  
  private processMessageQueue(): void {
    const queue = [...this.messageQueue];
    this.messageQueue = [];
    
    queue.forEach(message => {
      this.send(message);
    });
  }
  
  private updateConnectionState(updates: Partial<ConnectionState>): void {
    this.connectionState = {
      ...this.connectionState,
      ...updates
    };
    
    // Notify state change handlers
    this.stateChangeHandlers.forEach(handler => {
      handler(this.connectionState);
    });
  }
  
  private cleanup(): void {
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      
      if (this.ws.readyState === WEBSOCKET_READY_STATES.OPEN) {
        this.ws.close(1000, 'Client disconnect');
      }
      
      this.ws = null;
    }
    
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }
}

// === WEBSOCKET HOOK ===

export const createWebSocketService = (config: ConnectionConfig): CollaborativeWebSocketService => {
  return new CollaborativeWebSocketService(config);
};

export default CollaborativeWebSocketService;