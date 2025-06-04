/**
 * Composant d'édition collaborative principal - M&A Intelligence Platform
 * Sprint 5 - Interface de test pour édition collaborative Y.js + Tiptap
 */

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Avatar } from '@/components/ui/avatar';
import { 
  Users, 
  Wifi, 
  WifiOff, 
  Save, 
  MessageSquare, 
  Eye,
  Settings,
  Activity,
  Clock,
  AlertCircle,
  CheckCircle,
  Loader
} from 'lucide-react';

import { useCollaborativeEditor } from '../hooks/useCollaborativeEditor';
import type { EditorConfig, CollaboratorInfo, Comment } from '../types';

interface CollaborativeEditorProps {
  documentId: string;
  userId: string;
  username: string;
  avatar?: string;
  className?: string;
}

const CollaborativeEditor: React.FC<CollaborativeEditorProps> = ({
  documentId,
  userId,
  username,
  avatar,
  className = ""
}) => {
  // Configuration de l'éditeur
  const editorConfig: EditorConfig = {
    documentId,
    userId,
    username,
    avatar,
    websocketUrl: `ws://localhost:8000/api/v1/collaborative/ws/yjs/${documentId}`,
    fallbackPollingUrl: `/api/v1/collaborative/document/polling`,
    autoSave: true,
    autoSaveInterval: 3000
  };

  // Hook principal d'édition collaborative
  const {
    editorState,
    connectionState,
    collaborators,
    comments,
    presence,
    connect,
    disconnect,
    saveDocument,
    addComment,
    resolveComment,
    editorRef,
    isLoading,
    isSaving,
    isConnecting,
    error
  } = useCollaborativeEditor(editorConfig);

  // États locaux pour l'interface
  const [showSettings, setShowSettings] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [newCommentContent, setNewCommentContent] = useState('');
  const [selectedPosition, setSelectedPosition] = useState<number>(0);
  const [mockContent, setMockContent] = useState('<p>Bienvenue dans l\'éditeur collaboratif !</p><p>Commencez à taper pour tester la collaboration...</p>');

  // Auto-connexion au chargement
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Gestion des erreurs de connexion
  const getConnectionIcon = () => {
    switch (connectionState.status) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />;
      case 'connecting':
      case 'reconnecting':
        return <Loader className="h-4 w-4 text-yellow-500 animate-spin" />;
      case 'disconnected':
      case 'error':
        return <WifiOff className="h-4 w-4 text-red-500" />;
      default:
        return <WifiOff className="h-4 w-4 text-gray-500" />;
    }
  };

  const getConnectionStatusText = () => {
    if (connectionState.pollingFallback) {
      return `${connectionState.status} (polling)`;
    }
    return connectionState.status;
  };

  // Ajouter un commentaire
  const handleAddComment = async () => {
    if (!newCommentContent.trim()) return;
    
    try {
      await addComment(selectedPosition, newCommentContent);
      setNewCommentContent('');
    } catch (err) {
      console.error('Failed to add comment:', err);
    }
  };

  // Marquer un commentaire comme résolu
  const handleResolveComment = async (commentId: string) => {
    try {
      await resolveComment(commentId);
    } catch (err) {
      console.error('Failed to resolve comment:', err);
    }
  };

  // Simulation de contenu éditeur
  const handleContentChange = (newContent: string) => {
    setMockContent(newContent);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="h-8 w-8 animate-spin" />
        <span className="ml-2">Initialisation de l'éditeur collaboratif...</span>
      </div>
    );
  }

  return (
    <div className={`collaborative-editor ${className}`}>
      {/* Header avec status et collaborateurs */}
      <Card className="mb-4">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Éditeur Collaboratif
              <Badge variant={editorState.isConnected ? "default" : "secondary"}>
                {getConnectionStatusText()}
              </Badge>
            </CardTitle>
            
            <div className="flex items-center gap-2">
              {/* Indicateur de connexion */}
              <div className="flex items-center gap-1">
                {getConnectionIcon()}
                <span className="text-sm text-muted-foreground">
                  {presence.activeCount} utilisateur{presence.activeCount > 1 ? 's' : ''}
                </span>
              </div>
              
              {/* Collaborateurs */}
              <div className="flex items-center gap-1">
                <Users className="h-4 w-4" />
                <div className="flex -space-x-2">
                  {/* Utilisateur actuel */}
                  <Avatar className="h-8 w-8 border-2 border-background">
                    <div className="h-full w-full bg-blue-500 flex items-center justify-center text-white text-xs">
                      {username.charAt(0).toUpperCase()}
                    </div>
                  </Avatar>
                  
                  {/* Autres collaborateurs */}
                  {collaborators.slice(0, 3).map((collaborator) => (
                    <Avatar 
                      key={collaborator.userId} 
                      className="h-8 w-8 border-2 border-background"
                    >
                      <div 
                        className="h-full w-full flex items-center justify-center text-white text-xs"
                        style={{ backgroundColor: collaborator.color }}
                      >
                        {collaborator.username.charAt(0).toUpperCase()}
                      </div>
                    </Avatar>
                  ))}
                  
                  {collaborators.length > 3 && (
                    <Avatar className="h-8 w-8 border-2 border-background">
                      <div className="h-full w-full bg-gray-500 flex items-center justify-center text-white text-xs">
                        +{collaborators.length - 3}
                      </div>
                    </Avatar>
                  )}
                </div>
              </div>
              
              {/* Actions */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowComments(!showComments)}
                className="relative"
              >
                <MessageSquare className="h-4 w-4" />
                {comments.filter(c => !c.isResolved).length > 0 && (
                  <Badge 
                    variant="destructive" 
                    className="absolute -top-2 -right-2 h-5 w-5 p-0 text-xs"
                  >
                    {comments.filter(c => !c.isResolved).length}
                  </Badge>
                )}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={saveDocument}
                disabled={isSaving || !editorState.hasUnsavedChanges}
              >
                {isSaving ? (
                  <Loader className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
          
          {/* Informations sur le document */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>Document: {documentId}</span>
            {editorState.lastSaved && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Sauvé à {new Date(editorState.lastSaved).toLocaleTimeString()}
              </span>
            )}
            {editorState.hasUnsavedChanges && (
              <Badge variant="outline" className="text-yellow-600">
                Modifications non sauvées
              </Badge>
            )}
          </div>
        </CardHeader>
      </Card>

      {/* Alertes d'erreur */}
      {error && (
        <Alert className="mb-4" variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <div>
            <strong>{error.code}</strong>: {error.message}
            {error.recoverable && (
              <Button 
                variant="outline" 
                size="sm" 
                className="ml-2"
                onClick={connect}
              >
                Réessayer
              </Button>
            )}
          </div>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Zone d'édition principale */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Zone d'édition</CardTitle>
            </CardHeader>
            <CardContent>
              {/* Simulation de l'éditeur Tiptap */}
              <div className="border rounded-md min-h-96 p-4 focus-within:ring-2 focus-within:ring-blue-500">
                <Textarea
                  value={mockContent}
                  onChange={(e) => handleContentChange(e.target.value)}
                  placeholder="Commencez à taper pour tester l'édition collaborative..."
                  className="min-h-80 border-none resize-none focus:ring-0"
                  onSelect={(e) => setSelectedPosition(e.currentTarget.selectionStart)}
                />
              </div>
              
              {/* Indicateurs de collaboration */}
              <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                <div className="flex items-center gap-4">
                  <span>Position: {selectedPosition}</span>
                  <span>Caractères: {mockContent.length}</span>
                </div>
                <div className="flex items-center gap-2">
                  {collaborators.map((collaborator) => (
                    <Badge 
                      key={collaborator.userId}
                      variant="outline"
                      style={{ borderColor: collaborator.color }}
                      className="text-xs"
                    >
                      <Eye className="h-3 w-3 mr-1" />
                      {collaborator.username}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Panneau latéral */}
        <div className="space-y-4">
          {/* Collaborateurs */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Users className="h-4 w-4" />
                Collaborateurs ({presence.totalCount})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Utilisateur actuel */}
              <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                <Avatar className="h-6 w-6">
                  <div className="h-full w-full bg-blue-500 flex items-center justify-center text-white text-xs">
                    {username.charAt(0).toUpperCase()}
                  </div>
                </Avatar>
                <span className="text-sm font-medium">{username} (vous)</span>
                <Badge variant="default" className="text-xs">En ligne</Badge>
              </div>
              
              {/* Autres collaborateurs */}
              {collaborators.map((collaborator) => (
                <div key={collaborator.userId} className="flex items-center gap-2 p-2 rounded">
                  <Avatar className="h-6 w-6">
                    <div 
                      className="h-full w-full flex items-center justify-center text-white text-xs"
                      style={{ backgroundColor: collaborator.color }}
                    >
                      {collaborator.username.charAt(0).toUpperCase()}
                    </div>
                  </Avatar>
                  <span className="text-sm">{collaborator.username}</span>
                  <Badge 
                    variant={collaborator.isActive ? "default" : "secondary"}
                    className="text-xs"
                  >
                    {collaborator.isActive ? "En ligne" : "Hors ligne"}
                  </Badge>
                </div>
              ))}
              
              {collaborators.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Aucun autre collaborateur connecté
                </p>
              )}
            </CardContent>
          </Card>

          {/* Commentaires */}
          {showComments && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Commentaires ({comments.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Nouveau commentaire */}
                <div className="space-y-2">
                  <Input
                    placeholder="Position du commentaire"
                    type="number"
                    value={selectedPosition}
                    onChange={(e) => setSelectedPosition(parseInt(e.target.value) || 0)}
                  />
                  <Textarea
                    placeholder="Ajouter un commentaire..."
                    value={newCommentContent}
                    onChange={(e) => setNewCommentContent(e.target.value)}
                    rows={2}
                  />
                  <Button 
                    size="sm" 
                    onClick={handleAddComment}
                    disabled={!newCommentContent.trim()}
                    className="w-full"
                  >
                    Ajouter commentaire
                  </Button>
                </div>
                
                {/* Liste des commentaires */}
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {comments.map((comment) => (
                    <div 
                      key={comment.id} 
                      className={`p-2 rounded border ${comment.isResolved ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1">
                          <Avatar className="h-4 w-4">
                            <div 
                              className="h-full w-full flex items-center justify-center text-white text-xs"
                              style={{ backgroundColor: comment.author.color }}
                            >
                              {comment.author.username.charAt(0).toUpperCase()}
                            </div>
                          </Avatar>
                          <span className="text-xs font-medium">{comment.author.username}</span>
                          <Badge variant="outline" className="text-xs">
                            @{comment.position}
                          </Badge>
                        </div>
                        {!comment.isResolved && (
                          <Button 
                            size="sm" 
                            variant="ghost"
                            onClick={() => handleResolveComment(comment.id)}
                            className="h-6 w-6 p-0"
                          >
                            <CheckCircle className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">{comment.content}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(comment.createdAt).toLocaleTimeString()}
                        {comment.isResolved && " - Résolu"}
                      </p>
                    </div>
                  ))}
                  
                  {comments.length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      Aucun commentaire
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Paramètres de connexion */}
          {showSettings && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Paramètres
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <Badge variant={editorState.isConnected ? "default" : "secondary"}>
                      {connectionState.status}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>WebSocket:</span>
                    <span>{connectionState.websocketAvailable ? "Disponible" : "Indisponible"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Fallback:</span>
                    <span>{connectionState.pollingFallback ? "Polling" : "WebSocket"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tentatives:</span>
                    <span>{connectionState.reconnectAttempts}</span>
                  </div>
                  {connectionState.lastConnected && (
                    <div className="flex justify-between">
                      <span>Dernière connexion:</span>
                      <span>{new Date(connectionState.lastConnected).toLocaleTimeString()}</span>
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <Button 
                    size="sm" 
                    onClick={connect} 
                    disabled={isConnecting}
                    className="w-full"
                  >
                    {isConnecting ? (
                      <>
                        <Loader className="h-3 w-3 mr-1 animate-spin" />
                        Connexion...
                      </>
                    ) : (
                      'Reconnecter'
                    )}
                  </Button>
                  <Button 
                    size="sm" 
                    variant="outline" 
                    onClick={disconnect}
                    className="w-full"
                  >
                    Déconnecter
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default CollaborativeEditor;