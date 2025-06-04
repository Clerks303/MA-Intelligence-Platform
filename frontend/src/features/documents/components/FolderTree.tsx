/**
 * Composant Arbre de Dossiers - M&A Intelligence Platform
 * Sprint 3 - Navigation hiérarchique
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Button } from '../../../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Input } from '../../../components/ui/input';
import { cn } from '../../../lib/utils';
import { Folder, TreeNode } from '../types';
import { useFolderTree, useFolderMutations } from '../hooks/useDocuments';
import { useDocumentNavigation, useDocumentStore } from '../stores/documentStore';

interface FolderTreeProps {
  className?: string;
  onFolderSelect?: (folderId: string) => void;
  showCreateButton?: boolean;
  compactMode?: boolean;
}

interface TreeNodeItemProps {
  node: TreeNode;
  level: number;
  isSelected: boolean;
  onSelect: (folderId: string) => void;
  onToggle: (nodeId: string) => void;
  onCreateSubfolder?: (parentId: string) => void;
  compactMode?: boolean;
}

// Composant pour un item de l'arbre
const TreeNodeItem: React.FC<TreeNodeItemProps> = ({
  node,
  level,
  isSelected,
  onSelect,
  onToggle,
  onCreateSubfolder,
  compactMode = false,
}) => {
  const [isCreatingSubfolder, setIsCreatingSubfolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const { createFolder } = useFolderMutations();

  const handleCreateSubfolder = useCallback(() => {
    if (newFolderName.trim()) {
      createFolder({ name: newFolderName.trim(), parentId: node.id });
      setNewFolderName('');
      setIsCreatingSubfolder(false);
    }
  }, [newFolderName, node.id, createFolder]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreateSubfolder();
    } else if (e.key === 'Escape') {
      setIsCreatingSubfolder(false);
      setNewFolderName('');
    }
  }, [handleCreateSubfolder]);

  const hasChildren = node.children && node.children.length > 0;
  const indent = level * (compactMode ? 12 : 16);

  return (
    <div className="select-none">
      {/* Noeud principal */}
      <div
        className={cn(
          "flex items-center gap-1 py-1 px-2 rounded cursor-pointer group transition-colors",
          isSelected 
            ? "bg-ma-blue-100 text-ma-blue-900" 
            : "hover:bg-ma-slate-100",
          compactMode ? "text-sm" : "text-sm"
        )}
        style={{ paddingLeft: `${indent + 8}px` }}
        onClick={() => onSelect(node.id)}
      >
        {/* Icône expand/collapse */}
        {hasChildren && (
          <Button
            variant="ghost"
            size="icon"
            className="h-4 w-4 p-0 hover:bg-ma-slate-200"
            onClick={(e) => {
              e.stopPropagation();
              onToggle(node.id);
            }}
          >
            <svg 
              className={cn(
                "h-3 w-3 transition-transform",
                node.isExpanded ? "rotate-90" : ""
              )} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Button>
        )}

        {/* Icône dossier */}
        <div className="flex-shrink-0">
          <svg 
            className={cn(
              "h-4 w-4",
              hasChildren && node.isExpanded ? "text-ma-blue-600" : "text-ma-slate-600"
            )} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" 
            />
          </svg>
        </div>

        {/* Nom du dossier */}
        <span className={cn(
          "flex-1 truncate",
          isSelected ? "font-medium" : ""
        )}>
          {node.name}
        </span>

        {/* Nombre de documents */}
        {node.metadata && (node.metadata as Folder).documentCount > 0 && (
          <span className="text-xs text-ma-slate-500 bg-ma-slate-100 px-1.5 py-0.5 rounded">
            {(node.metadata as Folder).documentCount}
          </span>
        )}

        {/* Actions au survol */}
        {onCreateSubfolder && (
          <Button
            variant="ghost"
            size="icon"
            className="h-4 w-4 p-0 opacity-0 group-hover:opacity-100 hover:bg-ma-blue-100"
            onClick={(e) => {
              e.stopPropagation();
              setIsCreatingSubfolder(true);
            }}
            title="Créer sous-dossier"
          >
            <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </Button>
        )}
      </div>

      {/* Formulaire création sous-dossier */}
      {isCreatingSubfolder && (
        <div 
          className="mt-1 mb-2"
          style={{ paddingLeft: `${indent + 24}px` }}
        >
          <Input
            size="sm"
            placeholder="Nom du dossier"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => {
              if (!newFolderName.trim()) {
                setIsCreatingSubfolder(false);
              }
            }}
            autoFocus
            className="text-xs"
          />
        </div>
      )}

      {/* Enfants récursifs */}
      {hasChildren && node.isExpanded && (
        <div className="space-y-0.5">
          {node.children.map(child => (
            <TreeNodeItem
              key={child.id}
              node={child}
              level={level + 1}
              isSelected={isSelected && child.id === node.id}
              onSelect={onSelect}
              onToggle={onToggle}
              onCreateSubfolder={onCreateSubfolder}
              compactMode={compactMode}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Recherche dans l'arbre
const useTreeSearch = (tree: TreeNode[], searchQuery: string) => {
  return useMemo(() => {
    if (!searchQuery.trim()) return tree;

    const filterTree = (nodes: TreeNode[]): TreeNode[] => {
      return nodes.reduce((acc, node) => {
        const matchesSearch = node.name.toLowerCase().includes(searchQuery.toLowerCase());
        const filteredChildren = filterTree(node.children);
        
        if (matchesSearch || filteredChildren.length > 0) {
          acc.push({
            ...node,
            children: filteredChildren,
            isExpanded: filteredChildren.length > 0 ? true : node.isExpanded,
          });
        }
        
        return acc;
      }, [] as TreeNode[]);
    };

    return filterTree(tree);
  }, [tree, searchQuery]);
};

export const FolderTree: React.FC<FolderTreeProps> = ({
  className,
  onFolderSelect,
  showCreateButton = true,
  compactMode = false,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isCreatingRoot, setIsCreatingRoot] = useState(false);
  const [newRootName, setNewRootName] = useState('');

  const { folderTree, isLoading, error } = useFolderTree();
  const { currentFolderId, navigateToFolder } = useDocumentNavigation();
  const { expandTreeNode, collapseTreeNode, toggleTreeNode } = useDocumentStore();
  const { createFolder } = useFolderMutations();

  // Transformer les dossiers en TreeNodes
  const treeNodes = useMemo(() => {
    const buildTreeNodes = (folders: Folder[], level = 0): TreeNode[] => {
      return folders.map(folder => ({
        id: folder.id,
        name: folder.name,
        type: 'folder' as const,
        parentId: folder.parentId,
        level,
        isExpanded: folder.isExpanded || false,
        isLoading: folder.isLoading || false,
        children: buildTreeNodes(folder.children, level + 1),
        path: folder.path,
        metadata: folder,
      }));
    };

    return buildTreeNodes(folderTree);
  }, [folderTree]);

  // Filtrage par recherche
  const filteredTree = useTreeSearch(treeNodes, searchQuery);

  const handleFolderSelect = useCallback((folderId: string) => {
    const findFolder = (nodes: TreeNode[]): Folder | null => {
      for (const node of nodes) {
        if (node.id === folderId && node.metadata) {
          return node.metadata as Folder;
        }
        const found = findFolder(node.children);
        if (found) return found;
      }
      return null;
    };

    const folder = findFolder(treeNodes);
    if (folder) {
      navigateToFolder(folder.id, folder.name, folder.path);
      onFolderSelect?.(folderId);
    }
  }, [treeNodes, navigateToFolder, onFolderSelect]);

  const handleCreateRoot = useCallback(() => {
    if (newRootName.trim()) {
      createFolder({ name: newRootName.trim() });
      setNewRootName('');
      setIsCreatingRoot(false);
    }
  }, [newRootName, createFolder]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader className={compactMode ? "py-3" : undefined}>
          <CardTitle className={compactMode ? "text-base" : undefined}>
            Dossiers
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-6 bg-ma-slate-200 rounded"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Dossiers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <svg className="w-8 h-8 text-ma-red-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-ma-red-600">Erreur chargement dossiers</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className={cn(compactMode && "py-3")}>
        <div className="flex items-center justify-between">
          <CardTitle className={cn(compactMode && "text-base")}>
            Dossiers
          </CardTitle>
          {showCreateButton && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setIsCreatingRoot(true)}
              title="Créer dossier"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </Button>
          )}
        </div>
        
        {/* Recherche */}
        <div className="pt-2">
          <Input
            size="sm"
            placeholder="Rechercher dossier..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            icon={
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            }
          />
        </div>
      </CardHeader>

      <CardContent className={cn("py-2", compactMode && "py-1")}>
        {/* Formulaire création dossier racine */}
        {isCreatingRoot && (
          <div className="mb-3">
            <Input
              size="sm"
              placeholder="Nom du dossier"
              value={newRootName}
              onChange={(e) => setNewRootName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleCreateRoot();
                } else if (e.key === 'Escape') {
                  setIsCreatingRoot(false);
                  setNewRootName('');
                }
              }}
              onBlur={() => {
                if (!newRootName.trim()) {
                  setIsCreatingRoot(false);
                }
              }}
              autoFocus
            />
          </div>
        )}

        {/* Arbre de dossiers */}
        <div className="space-y-0.5 max-h-96 overflow-y-auto custom-scrollbar">
          {/* Racine "Tous les documents" */}
          <div
            className={cn(
              "flex items-center gap-2 py-1 px-2 rounded cursor-pointer transition-colors",
              !currentFolderId 
                ? "bg-ma-blue-100 text-ma-blue-900 font-medium" 
                : "hover:bg-ma-slate-100"
            )}
            onClick={() => handleFolderSelect('root')}
          >
            <svg className="h-4 w-4 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            <span className="flex-1">Tous les documents</span>
          </div>

          {/* Arbre des dossiers */}
          {filteredTree.map(node => (
            <TreeNodeItem
              key={node.id}
              node={node}
              level={0}
              isSelected={currentFolderId === node.id}
              onSelect={handleFolderSelect}
              onToggle={toggleTreeNode}
              onCreateSubfolder={showCreateButton ? () => {} : undefined}
              compactMode={compactMode}
            />
          ))}

          {/* Message si aucun dossier */}
          {filteredTree.length === 0 && !searchQuery && (
            <div className="text-center py-4 text-ma-slate-500">
              <svg className="w-8 h-8 mx-auto mb-2 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <p className="text-sm">Aucun dossier</p>
            </div>
          )}

          {/* Message si aucun résultat de recherche */}
          {filteredTree.length === 0 && searchQuery && (
            <div className="text-center py-4 text-ma-slate-500">
              <svg className="w-8 h-8 mx-auto mb-2 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p className="text-sm">Aucun dossier trouvé pour "{searchQuery}"</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};