/**
 * Layout Dashboard avec Drag & Drop - M&A Intelligence Platform
 * Sprint 2 - Layout responsive avec widgets réorganisables
 */

import React, { useState, useCallback } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import {
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { cn } from '../../../lib/utils';
import { DashboardWidget, DashboardLayout as DashboardLayoutType } from '../types';

interface DashboardLayoutProps {
  layout: DashboardLayoutType;
  onLayoutChange?: (layout: DashboardLayoutType) => void;
  isEditing?: boolean;
  onToggleEdit?: () => void;
  children: React.ReactNode;
  className?: string;
}

interface WidgetContainerProps {
  widget: DashboardWidget;
  isEditing: boolean;
  onRemove?: (widgetId: string) => void;
  onToggleVisibility?: (widgetId: string) => void;
  children: React.ReactNode;
}

// Container pour widget individuel avec drag & drop
const SortableWidget: React.FC<WidgetContainerProps> = ({
  widget,
  isEditing,
  onRemove,
  onToggleVisibility,
  children,
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: widget.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  if (!widget.visible) {
    return null;
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "relative group",
        isEditing && "ring-2 ring-ma-blue-200 ring-opacity-50 hover:ring-ma-blue-400",
        isDragging && "z-50"
      )}
    >
      {/* Overlay de contrôle en mode édition */}
      {isEditing && (
        <div className="absolute top-2 right-2 z-10 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Handle pour drag & drop */}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 bg-white/90 hover:bg-white shadow-sm"
            {...attributes}
            {...listeners}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
            </svg>
          </Button>

          {/* Toggle visibilité */}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 bg-white/90 hover:bg-white shadow-sm"
            onClick={() => onToggleVisibility?.(widget.id)}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </Button>

          {/* Supprimer widget */}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 bg-white/90 hover:bg-white hover:text-ma-red-600 shadow-sm"
            onClick={() => onRemove?.(widget.id)}
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Button>
        </div>
      )}

      {children}
    </div>
  );
};

// Layout principal du dashboard
export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  layout,
  onLayoutChange,
  isEditing = false,
  onToggleEdit,
  children,
  className,
}) => {
  const [widgets, setWidgets] = useState(layout.widgets);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Démarrer le drag après 8px de mouvement
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event;

    if (active.id !== over?.id) {
      setWidgets((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over?.id);

        const newWidgets = arrayMove(items, oldIndex, newIndex);
        
        // Mettre à jour les positions Y dans la grille
        const updatedWidgets = newWidgets.map((widget, index) => ({
          ...widget,
          gridPosition: {
            ...widget.gridPosition,
            y: index, // Nouvel ordre
          },
        }));

        // Notifier le parent du changement
        if (onLayoutChange) {
          onLayoutChange({
            ...layout,
            widgets: updatedWidgets,
            updatedAt: new Date(),
          });
        }

        return updatedWidgets;
      });
    }
  }, [layout, onLayoutChange]);

  const handleRemoveWidget = useCallback((widgetId: string) => {
    const updatedWidgets = widgets.filter(w => w.id !== widgetId);
    setWidgets(updatedWidgets);
    
    if (onLayoutChange) {
      onLayoutChange({
        ...layout,
        widgets: updatedWidgets,
        updatedAt: new Date(),
      });
    }
  }, [widgets, layout, onLayoutChange]);

  const handleToggleVisibility = useCallback((widgetId: string) => {
    const updatedWidgets = widgets.map(w => 
      w.id === widgetId ? { ...w, visible: !w.visible } : w
    );
    setWidgets(updatedWidgets);
    
    if (onLayoutChange) {
      onLayoutChange({
        ...layout,
        widgets: updatedWidgets,
        updatedAt: new Date(),
      });
    }
  }, [widgets, layout, onLayoutChange]);

  const visibleWidgets = widgets.filter(w => w.visible);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header du dashboard avec contrôles */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ma-slate-900">
            {layout.name}
          </h1>
          <p className="text-sm text-ma-slate-600">
            Dernière mise à jour: {layout.updatedAt.toLocaleTimeString('fr-FR')}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Mode édition */}
          <Button
            variant={isEditing ? "ma" : "outline"}
            size="sm"
            onClick={onToggleEdit}
            className="flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            {isEditing ? 'Terminer' : 'Modifier'}
          </Button>

          {/* Actualiser */}
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Actualiser
          </Button>
        </div>
      </div>

      {/* Message d'aide en mode édition */}
      {isEditing && (
        <Card className="border-ma-blue-200 bg-ma-blue-50">
          <CardContent className="py-3">
            <p className="text-sm text-ma-blue-800">
              <span className="font-medium">Mode édition activé:</span> Glissez-déposez les widgets pour les réorganiser. 
              Utilisez les contrôles en survol pour masquer ou supprimer des widgets.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Widgets avec drag & drop */}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={visibleWidgets.map(w => w.id)}
          strategy={verticalListSortingStrategy}
        >
          <div className="space-y-6">
            {React.Children.map(children, (child, index) => {
              const widget = visibleWidgets[index];
              if (!widget) return null;

              return (
                <SortableWidget
                  key={widget.id}
                  widget={widget}
                  isEditing={isEditing}
                  onRemove={handleRemoveWidget}
                  onToggleVisibility={handleToggleVisibility}
                >
                  {child}
                </SortableWidget>
              );
            })}
          </div>
        </SortableContext>
      </DndContext>

      {/* Widgets masqués (en mode édition) */}
      {isEditing && widgets.some(w => !w.visible) && (
        <Card className="border-ma-slate-200 bg-ma-slate-50">
          <CardHeader>
            <CardTitle className="text-base text-ma-slate-700">
              Widgets masqués
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {widgets
                .filter(w => !w.visible)
                .map(widget => (
                  <Button
                    key={widget.id}
                    variant="outline"
                    size="sm"
                    onClick={() => handleToggleVisibility(widget.id)}
                    className="flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    {widget.title}
                  </Button>
                ))
              }
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

// Layout pour grille responsive (alternative sans drag & drop)
interface ResponsiveGridProps {
  children: React.ReactNode;
  className?: string;
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({ children, className }) => {
  return (
    <div className={cn(
      "grid gap-6",
      "grid-cols-1",
      "lg:grid-cols-2",
      "xl:grid-cols-3",
      "2xl:grid-cols-4",
      className
    )}>
      {children}
    </div>
  );
};

// Layout pour widgets en colonnes
interface ColumnsLayoutProps {
  leftColumn: React.ReactNode;
  rightColumn: React.ReactNode;
  className?: string;
}

export const ColumnsLayout: React.FC<ColumnsLayoutProps> = ({ 
  leftColumn, 
  rightColumn, 
  className 
}) => {
  return (
    <div className={cn("grid grid-cols-1 lg:grid-cols-3 gap-6", className)}>
      <div className="lg:col-span-2 space-y-6">
        {leftColumn}
      </div>
      <div className="space-y-6">
        {rightColumn}
      </div>
    </div>
  );
};