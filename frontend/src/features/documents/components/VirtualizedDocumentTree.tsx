/**
 * Arbre de Navigation Virtualisé - M&A Intelligence Platform
 * Sprint 3 - Navigation arbre avec virtualisation et recherche sémantique
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { FixedSizeList as List } from 'react-window';
import { Input } from '../../../components/ui/input';
import { Button } from '../../../components/ui/button';
import { Badge } from '../../../components/ui/badge';
import { Card, CardHeader, CardContent } from '../../../components/ui/card';
import { 
  Search, 
  Folder, 
  FolderOpen, 
  File, 
  ChevronRight, 
  ChevronDown,
  Filter,
  SortAsc,
  MoreHorizontal,
  Eye,
  Download,
  Edit,
  Trash2
} from 'lucide-react';
import { cn } from '../../../lib/utils';

import { 
  Document, 
  DocumentFilters,
  BackendDocumentType,
  BackendAccessLevel,
  FrontendDocumentType
} from '../types';
import { useSemanticSearch, useVirtualizedDocuments } from '../hooks/useAdvancedDocuments';
import { useDocuments } from '../hooks/useDocuments';

interface TreeNode {
  id: string;
  name: string;
  type: 'folder' | 'document';
  level: number;
  isExpanded: boolean;
  document?: Document;
  children?: TreeNode[];
  parentId?: string;
}

interface VirtualizedDocumentTreeProps {
  documents: Document[];
  onSelectDocument?: (document: Document) => void;
  onDocumentAction?: (action: string, document: Document) => void;
  filters?: DocumentFilters;
  onFiltersChange?: (filters: DocumentFilters) => void;
  height?: number;
  enableSearch?: boolean;
  enableVirtualization?: boolean;
}

interface TreeItemProps {
  node: TreeNode;
  style: React.CSSProperties;
  onToggle: (nodeId: string) => void;
  onSelect: (node: TreeNode) => void;
  onAction: (action: string, document: Document) => void;
  searchTerm: string;
  isSelected: boolean;
}

// Utilitaires
const getDocumentTypeIcon = (documentType: BackendDocumentType) => {
  const icons: Record<BackendDocumentType, string> = {
    financial: '💰',
    legal: '⚖️',
    due_diligence: '🔍',
    communication: '💬',
    technical: '🔧',
    hr: '👥',
    commercial: '📈',
    other: '📄',
  };
  return icons[documentType] || '📄';
};

const getAccessLevelColor = (accessLevel: BackendAccessLevel) => {
  const colors: Record<BackendAccessLevel, string> = {
    public: 'bg-green-100 text-green-800',
    internal: 'bg-blue-100 text-blue-800',
    confidential: 'bg-orange-100 text-orange-800',
    restricted: 'bg-red-100 text-red-800',
  };
  return colors[accessLevel] || 'bg-gray-100 text-gray-800';
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const highlightText = (text: string, searchTerm: string) => {
  if (!searchTerm.trim()) return text;
  
  const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);
  
  return parts.map((part, index) => 
    regex.test(part) ? (
      <mark key={index} className="bg-yellow-200 text-yellow-900 rounded px-1">
        {part}
      </mark>
    ) : part
  );
};

// Composant TreeItem
const TreeItem: React.FC<TreeItemProps> = ({
  node,
  style,
  onToggle,
  onSelect,
  onAction,
  searchTerm,
  isSelected,
}) => {
  const [showActions, setShowActions] = useState(false);

  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onToggle(node.id);
  }, [node.id, onToggle]);

  const handleSelect = useCallback(() => {
    onSelect(node);
  }, [node, onSelect]);

  const handleAction = useCallback((action: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.document) {
      onAction(action, node.document);
    }
  }, [node.document, onAction]);

  const isFolder = node.type === 'folder';
  const document = node.document;

  return (
    <div
      style={style}
      className={cn(
        "flex items-center gap-2 px-2 py-1 hover:bg-gray-50 cursor-pointer group relative",
        isSelected && "bg-blue-50 border-l-2 border-l-blue-500",
        "border-b border-gray-100"
      )}
      onClick={handleSelect}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Indentation */}
      <div style={{ width: node.level * 20 }} />

      {/* Toggle button pour dossiers */}
      {isFolder && (
        <Button
          variant="ghost"
          size="icon"
          className="h-4 w-4 p-0"
          onClick={handleToggle}
        >
          {node.isExpanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </Button>
      )}

      {/* Icône */}
      <div className="flex-shrink-0">
        {isFolder ? (
          node.isExpanded ? (
            <FolderOpen className="h-4 w-4 text-blue-600" />
          ) : (
            <Folder className="h-4 w-4 text-blue-500" />
          )
        ) : (
          <div className="flex items-center">
            <File className="h-4 w-4 text-gray-500" />
            {document && (
              <span className="ml-1 text-xs">
                {getDocumentTypeIcon(document.document_type)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Nom et métadonnées */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn(
            "text-sm truncate",
            isSelected ? "font-medium text-blue-900" : "text-gray-900"
          )}>
            {highlightText(node.name, searchTerm)}
          </span>
          
          {/* Badges pour documents */}
          {document && (
            <div className="flex items-center gap-1">
              <Badge
                variant="secondary"
                className={cn("text-xs px-1 py-0", getAccessLevelColor(document.access_level))}
              >
                {document.access_level}
              </Badge>
              
              {document.view_count > 0 && (
                <Badge variant="outline" className="text-xs px-1 py-0">
                  {document.view_count} vues
                </Badge>
              )}
            </div>
          )}
        </div>

        {/* Informations supplémentaires pour documents */}
        {document && (
          <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
            <span>{formatFileSize(document.file_size)}</span>
            <span>•</span>
            <span>{document.document_type}</span>
            <span>•</span>
            <span>{new Date(document.created_at).toLocaleDateString()}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      {document && showActions && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => handleAction('view', e)}
            title="Voir"
          >
            <Eye className="h-3 w-3" />
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => handleAction('download', e)}
            title="Télécharger"
          >
            <Download className="h-3 w-3" />
          </Button>
          
          {document.canEdit && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={(e) => handleAction('edit', e)}
              title="Modifier"
            >
              <Edit className="h-3 w-3" />
            </Button>
          )}
          
          {document.canDelete && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-red-600 hover:text-red-700"
              onClick={(e) => handleAction('delete', e)}
              title="Supprimer"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
};

// Composant principal
export const VirtualizedDocumentTree: React.FC<VirtualizedDocumentTreeProps> = ({
  documents,
  onSelectDocument,
  onDocumentAction,
  filters = {},
  onFiltersChange,
  height = 600,
  enableSearch = true,
  enableVirtualization = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [localSearchTerm, setLocalSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Recherche sémantique
  const {
    query: semanticQuery,
    setQuery: setSemanticQuery,
    results: semanticResults,
    isLoading: isSearching,
    isSemanticEnabled,
    toggleSemanticMode,
  } = useSemanticSearch({
    autoSearch: true,
    minQueryLength: 2,
    debounceMs: 300,
  });

  // Construction de l'arbre
  const treeNodes = useMemo(() => {
    // Utiliser les résultats de recherche sémantique si disponibles
    const documentsToShow = semanticQuery.length >= 2 && semanticResults.length > 0
      ? semanticResults.map(result => result.metadata)
      : documents;

    // Grouper par dossier virtuel basé sur document_type
    const groupedDocs: Record<string, Document[]> = {};
    
    documentsToShow.forEach(doc => {
      const groupKey = doc.document_type;
      if (!groupedDocs[groupKey]) {
        groupedDocs[groupKey] = [];
      }
      groupedDocs[groupKey].push(doc);
    });

    // Créer les nœuds de l'arbre
    const nodes: TreeNode[] = [];
    
    Object.entries(groupedDocs).forEach(([docType, docs]) => {
      const folderId = `folder-${docType}`;
      const isExpanded = expandedNodes.has(folderId);
      
      // Nœud dossier
      const folderNode: TreeNode = {
        id: folderId,
        name: `${getDocumentTypeIcon(docType as BackendDocumentType)} ${docType} (${docs.length})`,
        type: 'folder',
        level: 0,
        isExpanded,
      };
      
      nodes.push(folderNode);
      
      // Si étendu, ajouter les documents
      if (isExpanded) {
        docs
          .filter(doc => {
            // Filtrage local
            if (localSearchTerm) {
              const searchLower = localSearchTerm.toLowerCase();
              return (
                doc.filename.toLowerCase().includes(searchLower) ||
                doc.title?.toLowerCase().includes(searchLower) ||
                doc.description?.toLowerCase().includes(searchLower) ||
                doc.tags.some(tag => tag.toLowerCase().includes(searchLower))
              );
            }
            return true;
          })
          .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
          .forEach(doc => {
            nodes.push({
              id: `doc-${doc.document_id}`,
              name: doc.title || doc.filename,
              type: 'document',
              level: 1,
              isExpanded: false,
              document: doc,
              parentId: folderId,
            });
          });
      }
    });

    return nodes;
  }, [documents, semanticResults, semanticQuery, expandedNodes, localSearchTerm]);

  // Virtualisation
  const { rowVirtualizer, virtualItems, totalSize, performanceMetrics } = enableVirtualization
    ? useVirtualizedDocuments(treeNodes, containerRef)
    : { rowVirtualizer: null, virtualItems: null, totalSize: null, performanceMetrics: null };

  // Actions
  const handleToggleNode = useCallback((nodeId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  }, []);

  const handleSelectNode = useCallback((node: TreeNode) => {
    setSelectedNodeId(node.id);
    if (node.document && onSelectDocument) {
      onSelectDocument(node.document);
    }
  }, [onSelectDocument]);

  const handleDocumentAction = useCallback((action: string, document: Document) => {
    if (onDocumentAction) {
      onDocumentAction(action, document);
    }
  }, [onDocumentAction]);

  // Effets
  useEffect(() => {
    // Étendre automatiquement le premier dossier
    if (treeNodes.length > 0 && expandedNodes.size === 0) {
      const firstFolder = treeNodes.find(node => node.type === 'folder');
      if (firstFolder) {
        setExpandedNodes(new Set([firstFolder.id]));
      }
    }
  }, [treeNodes, expandedNodes.size]);

  const totalDocuments = documents.length;
  const visibleDocuments = treeNodes.filter(node => node.type === 'document').length;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="space-y-3">
          {/* Recherche */}
          {enableSearch && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder={isSemanticEnabled ? "Recherche sémantique..." : "Recherche simple..."}
                    value={isSemanticEnabled ? semanticQuery : localSearchTerm}
                    onChange={(e) => {
                      if (isSemanticEnabled) {
                        setSemanticQuery(e.target.value);
                      } else {
                        setLocalSearchTerm(e.target.value);
                      }
                    }}
                    className="pl-10"
                  />
                  {isSearching && (
                    <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                      <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
                    </div>
                  )}
                </div>
                
                <Button
                  variant={isSemanticEnabled ? "default" : "outline"}
                  size="icon"
                  onClick={toggleSemanticMode}
                  title={isSemanticEnabled ? "Recherche sémantique active" : "Recherche simple active"}
                >
                  🧠
                </Button>
                
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setShowFilters(!showFilters)}
                >
                  <Filter className="h-4 w-4" />
                </Button>
              </div>

              {/* Mode de recherche */}
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Badge variant={isSemanticEnabled ? "default" : "secondary"} className="text-xs">
                  {isSemanticEnabled ? "Recherche IA" : "Recherche texte"}
                </Badge>
                {isSemanticEnabled && semanticResults.length > 0 && (
                  <span>{semanticResults.length} résultats pertinents</span>
                )}
              </div>
            </div>
          )}

          {/* Statistiques */}
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>
              {visibleDocuments} / {totalDocuments} documents
              {enableVirtualization && performanceMetrics && (
                <span className="ml-2 text-xs text-green-600">
                  (Mémoire optimisée: {performanceMetrics.memoryOptimization})
                </span>
              )}
            </span>
            
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setExpandedNodes(new Set())}
                className="text-xs"
              >
                Réduire tout
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  const folderNodes = treeNodes
                    .filter(node => node.type === 'folder')
                    .map(node => node.id);
                  setExpandedNodes(new Set(folderNodes));
                }}
                className="text-xs"
              >
                Étendre tout
              </Button>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden">
        <div ref={containerRef} className="h-full">
          {enableVirtualization && rowVirtualizer ? (
            // Mode virtualisé pour performance
            <div
              style={{
                height: totalSize,
                width: '100%',
                position: 'relative',
              }}
            >
              {virtualItems?.map((virtualItem) => {
                const node = treeNodes[virtualItem.index];
                if (!node) return null;

                return (
                  <TreeItem
                    key={node.id}
                    node={node}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: `${virtualItem.size}px`,
                      transform: `translateY(${virtualItem.start}px)`,
                    }}
                    onToggle={handleToggleNode}
                    onSelect={handleSelectNode}
                    onAction={handleDocumentAction}
                    searchTerm={isSemanticEnabled ? semanticQuery : localSearchTerm}
                    isSelected={selectedNodeId === node.id}
                  />
                );
              })}
            </div>
          ) : (
            // Mode standard
            <div className="space-y-0">
              {treeNodes.map((node) => (
                <TreeItem
                  key={node.id}
                  node={node}
                  style={{}}
                  onToggle={handleToggleNode}
                  onSelect={handleSelectNode}
                  onAction={handleDocumentAction}
                  searchTerm={isSemanticEnabled ? semanticQuery : localSearchTerm}
                  isSelected={selectedNodeId === node.id}
                />
              ))}
            </div>
          )}
          
          {treeNodes.length === 0 && (
            <div className="flex items-center justify-center h-40 text-gray-500">
              <div className="text-center">
                <File className="mx-auto h-8 w-8 mb-2" />
                <p>Aucun document trouvé</p>
                {(semanticQuery || localSearchTerm) && (
                  <p className="text-sm mt-1">Essayez de modifier votre recherche</p>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};