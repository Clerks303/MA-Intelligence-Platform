/**
 * Page de test pour édition collaborative - M&A Intelligence Platform
 * Sprint 5 - Interface complète de démonstration des fonctionnalités collaboratives
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  FileText, 
  Users, 
  Activity, 
  Settings,
  Plus,
  Clock,
  Globe,
  Wifi,
  MessageSquare,
  Eye,
  Monitor,
  Smartphone,
  Tablet,
  Info
} from 'lucide-react';

import CollaborativeEditor from './CollaborativeEditor';

interface TestDocument {
  id: string;
  title: string;
  createdAt: string;
  collaboratorCount: number;
  status: 'active' | 'archived';
}

interface TestUser {
  id: string;
  username: string;
  avatar?: string;
}

const CollaborativeTestPage: React.FC = () => {
  // État pour les documents de test
  const [documents] = useState<TestDocument[]>([
    {
      id: 'doc-1',
      title: 'Document de test collaboratif',
      createdAt: new Date().toISOString(),
      collaboratorCount: 2,
      status: 'active'
    },
    {
      id: 'doc-2', 
      title: 'Rapport M&A Intelligence',
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      collaboratorCount: 0,
      status: 'active'
    }
  ]);

  // État pour les utilisateurs de test
  const [testUsers] = useState<TestUser[]>([
    { id: 'user-1', username: 'Alice Martin' },
    { id: 'user-2', username: 'Bob Dupont' },
    { id: 'user-3', username: 'Claire Dubois' },
    { id: 'user-4', username: 'David Lopez' }
  ]);

  // État de l'interface
  const [selectedDocument, setSelectedDocument] = useState<string>('doc-1');
  const [selectedUser, setSelectedUser] = useState<TestUser>(testUsers[0]);
  const [showMultipleUsers, setShowMultipleUsers] = useState(false);

  // Créer un nouveau document de test
  const createTestDocument = () => {
    const newDoc: TestDocument = {
      id: `doc-${Date.now()}`,
      title: `Nouveau document ${new Date().toLocaleTimeString()}`,
      createdAt: new Date().toISOString(),
      collaboratorCount: 0,
      status: 'active'
    };
    
    documents.push(newDoc);
    setSelectedDocument(newDoc.id);
  };

  // Simuler différents scénarios
  const simulateScenario = (scenario: string) => {
    switch (scenario) {
      case 'multiple-users':
        setShowMultipleUsers(true);
        alert('Simulation: Plusieurs utilisateurs rejoignent le document');
        break;
      case 'network-issues':
        alert('Simulation: Problèmes réseau - fallback vers polling');
        break;
      case 'conflict-resolution':
        alert('Simulation: Résolution de conflits d\'édition');
        break;
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <FileText className="h-8 w-8" />
            Test Édition Collaborative
          </h1>
          <p className="text-muted-foreground mt-1">
            Démonstration des fonctionnalités Y.js + Tiptap + WebSocket
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Globe className="h-3 w-3" />
            Sprint 5
          </Badge>
          <Badge variant="default" className="flex items-center gap-1">
            <Wifi className="h-3 w-3" />
            Temps réel
          </Badge>
        </div>
      </div>

      {/* Information sur la démo */}
      <Alert>
        <Info className="h-4 w-4" />
        <div>
          <strong>Mode Démonstration</strong>
          <p className="text-sm mt-1">
            Cette interface utilise des mocks Y.js et Tiptap pour la démonstration. 
            L'architecture complète est en place et sera fonctionnelle une fois les dépendances installées.
          </p>
        </div>
      </Alert>

      <Tabs defaultValue="editor" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="editor" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Éditeur
          </TabsTrigger>
          <TabsTrigger value="users" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            Utilisateurs
          </TabsTrigger>
          <TabsTrigger value="scenarios" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Scénarios
          </TabsTrigger>
          <TabsTrigger value="monitoring" className="flex items-center gap-2">
            <Monitor className="h-4 w-4" />
            Monitoring
          </TabsTrigger>
        </TabsList>

        {/* Onglet Éditeur */}
        <TabsContent value="editor" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            {/* Sélection du document */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Documents
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {documents.map((doc) => (
                  <div 
                    key={doc.id}
                    className={`p-2 rounded cursor-pointer border ${
                      selectedDocument === doc.id 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedDocument(doc.id)}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{doc.title}</span>
                      <Badge variant={doc.status === 'active' ? 'default' : 'secondary'} className="text-xs">
                        {doc.status}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {new Date(doc.createdAt).toLocaleDateString()}
                      </span>
                      <span className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        {doc.collaboratorCount}
                      </span>
                    </div>
                  </div>
                ))}
                
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={createTestDocument}
                  className="w-full"
                >
                  <Plus className="h-3 w-3 mr-1" />
                  Nouveau document
                </Button>
              </CardContent>
            </Card>

            {/* Éditeur collaboratif */}
            <div className="lg:col-span-3">
              <CollaborativeEditor
                documentId={selectedDocument}
                userId={selectedUser.id}
                username={selectedUser.username}
                avatar={selectedUser.avatar}
              />
            </div>
          </div>
        </TabsContent>

        {/* Onglet Utilisateurs */}
        <TabsContent value="users" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {testUsers.map((user) => (
              <Card 
                key={user.id}
                className={`cursor-pointer ${
                  selectedUser.id === user.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedUser(user)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <div className="h-10 w-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-medium">
                      {user.username.charAt(0)}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">{user.username}</p>
                      <p className="text-sm text-muted-foreground">ID: {user.id}</p>
                    </div>
                    {selectedUser.id === user.id && (
                      <Badge variant="default">Actif</Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Utilisateur actuel</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 bg-blue-500 rounded-full flex items-center justify-center text-white font-medium">
                  {selectedUser.username.charAt(0)}
                </div>
                <div>
                  <p className="font-medium">{selectedUser.username}</p>
                  <p className="text-sm text-muted-foreground">
                    Connecté en tant que {selectedUser.username}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Onglet Scénarios */}
        <TabsContent value="scenarios" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Scénario 1: Utilisateurs multiples */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  Collaboration multi-utilisateur
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Simuler plusieurs utilisateurs qui éditent simultanément le document
                </p>
                <Button 
                  size="sm" 
                  onClick={() => simulateScenario('multiple-users')}
                  className="w-full"
                >
                  Simuler
                </Button>
              </CardContent>
            </Card>

            {/* Scénario 2: Problèmes réseau */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Wifi className="h-4 w-4" />
                  Fallback Polling
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Tester le basculement automatique vers le polling en cas de problème WebSocket
                </p>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => simulateScenario('network-issues')}
                  className="w-full"
                >
                  Simuler
                </Button>
              </CardContent>
            </Card>

            {/* Scénario 3: Résolution de conflits */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Résolution de conflits
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Tester la résolution automatique des conflits d'édition simultanée
                </p>
                <Button 
                  size="sm" 
                  variant="destructive"
                  onClick={() => simulateScenario('conflict-resolution')}
                  className="w-full"
                >
                  Simuler
                </Button>
              </CardContent>
            </Card>

            {/* Scénario 4: Commentaires temps réel */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Système de commentaires
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Tester l'ajout et la résolution de commentaires en temps réel
                </p>
                <Button 
                  size="sm" 
                  variant="secondary"
                  className="w-full"
                >
                  Ouvrir panneau
                </Button>
              </CardContent>
            </Card>

            {/* Scénario 5: Présence utilisateur */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Eye className="h-4 w-4" />
                  Awareness & Présence
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Tester l'affichage des curseurs et sélections des autres utilisateurs
                </p>
                <Button 
                  size="sm" 
                  variant="default"
                  className="w-full"
                >
                  Activer
                </Button>
              </CardContent>
            </Card>

            {/* Scénario 6: Responsive Design */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Smartphone className="h-4 w-4" />
                  Test Responsive
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Tester l'éditeur sur différentes tailles d'écran
                </p>
                <div className="flex gap-1">
                  <Button size="sm" variant="outline" className="flex-1">
                    <Smartphone className="h-3 w-3" />
                  </Button>
                  <Button size="sm" variant="outline" className="flex-1">
                    <Tablet className="h-3 w-3" />
                  </Button>
                  <Button size="sm" variant="outline" className="flex-1">
                    <Monitor className="h-3 w-3" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Onglet Monitoring */}
        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Métriques WebSocket */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Connexion WebSocket</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>État:</span>
                  <Badge variant="default">Connecté</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Latence:</span>
                  <span>45ms</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Messages envoyés:</span>
                  <span>127</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Messages reçus:</span>
                  <span>89</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Reconnexions:</span>
                  <span>0</span>
                </div>
              </CardContent>
            </Card>

            {/* Y.js Statistics */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Y.js Document</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Version:</span>
                  <span>v42</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Opérations:</span>
                  <span>156</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Taille document:</span>
                  <span>2.4 KB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Conflits résolus:</span>
                  <span>3</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Dernière sync:</span>
                  <span>il y a 2s</span>
                </div>
              </CardContent>
            </Card>

            {/* Performance */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Mémoire utilisée:</span>
                  <span>12 MB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Temps de rendu:</span>
                  <span>16ms</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Opérations/sec:</span>
                  <span>8.5</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Cache hit rate:</span>
                  <span>94%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Débits réseau:</span>
                  <span>1.2 KB/s</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Logs en temps réel */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Journal d'activité</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-50 rounded p-3 h-32 overflow-y-auto font-mono text-xs space-y-1">
                <div className="text-green-600">[{new Date().toLocaleTimeString()}] WebSocket connecté</div>
                <div className="text-blue-600">[{new Date().toLocaleTimeString()}] Y.js document initialisé</div>
                <div className="text-blue-600">[{new Date().toLocaleTimeString()}] Utilisateur {selectedUser.username} rejoint</div>
                <div className="text-yellow-600">[{new Date().toLocaleTimeString()}] Opération d'insertion détectée</div>
                <div className="text-green-600">[{new Date().toLocaleTimeString()}] Synchronisation Y.js réussie</div>
                <div className="text-blue-600">[{new Date().toLocaleTimeString()}] Awareness mis à jour - 2 utilisateurs actifs</div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CollaborativeTestPage;