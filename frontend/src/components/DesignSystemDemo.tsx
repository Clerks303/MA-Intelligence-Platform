/**
 * Design System Demo Component
 * Test et validation des composants ShadCN/UI
 */

import React, { useState } from 'react';
import { 
  Button, 
  Card, 
  CardHeader, 
  CardContent, 
  CardTitle, 
  CardDescription,
  StatsCard,
  Input, 
  SearchInput,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
  ConfirmDialog 
} from './ui';

export const DesignSystemDemo: React.FC = () => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [searchValue, setSearchValue] = useState('');

  return (
    <div className="p-8 space-y-8 bg-background text-foreground min-h-screen">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 text-ma-blue-700">
          M&A Intelligence Platform
        </h1>
        <p className="text-lg text-muted-foreground mb-8">
          Design System Demo - Sprint 1 Foundation
        </p>

        {/* Buttons Section */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">Buttons</h2>
          <div className="flex flex-wrap gap-3">
            <Button variant="default">Default</Button>
            <Button variant="ma">M&A Primary</Button>
            <Button variant="success">Success</Button>
            <Button variant="warning">Warning</Button>
            <Button variant="danger">Danger</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="link">Link</Button>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button size="xs">Extra Small</Button>
            <Button size="sm">Small</Button>
            <Button size="default">Default</Button>
            <Button size="lg">Large</Button>
            <Button size="icon">🚀</Button>
          </div>
        </section>

        {/* Cards Section */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">Cards</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatsCard
              title="Total Entreprises"
              value="1,247"
              change="+12% ce mois"
              trend="up"
            />
            <StatsCard
              title="En Prospection"
              value="342"
              change="+5% ce mois"
              trend="up"
            />
            <StatsCard
              title="Qualifiées"
              value="89"
              change="-2% ce mois"
              trend="down"
            />
            <StatsCard
              title="Contactées"
              value="156"
              change="Stable"
              trend="neutral"
            />
          </div>
          
          <Card className="max-w-md">
            <CardHeader>
              <CardTitle>Example Card</CardTitle>
              <CardDescription>
                Ceci est un exemple de card avec header et content.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p>Contenu de la card avec du texte d'exemple.</p>
            </CardContent>
          </Card>
        </section>

        {/* Inputs Section */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">Inputs</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
            <Input 
              placeholder="Input standard" 
            />
            <Input 
              placeholder="Input avec icône"
              icon={
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              }
            />
            <Input 
              placeholder="Input avec erreur"
              error="Ce champ est requis"
            />
            <SearchInput
              placeholder="Rechercher..."
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              onClear={() => setSearchValue('')}
              showClear={!!searchValue}
            />
          </div>
        </section>

        {/* Dialogs Section */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">Dialogs</h2>
          <div className="flex gap-3">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline">Ouvrir Dialog</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Exemple Dialog</DialogTitle>
                  <DialogDescription>
                    Ceci est un exemple de dialog avec du contenu personnalisé.
                  </DialogDescription>
                </DialogHeader>
                <div className="py-4">
                  <p>Contenu du dialog...</p>
                </div>
                <DialogFooter>
                  <Button variant="outline">Annuler</Button>
                  <Button variant="ma">Confirmer</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Button 
              variant="danger" 
              onClick={() => setIsConfirmOpen(true)}
            >
              Confirm Dialog
            </Button>
          </div>
        </section>

        {/* Theme Colors Demo */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">M&A Intelligence Colors</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <h3 className="font-medium">MA Blue</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="h-12 bg-ma-blue-400 rounded"></div>
                <div className="h-12 bg-ma-blue-600 rounded"></div>
                <div className="h-12 bg-ma-blue-800 rounded"></div>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">MA Green</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="h-12 bg-ma-green-400 rounded"></div>
                <div className="h-12 bg-ma-green-600 rounded"></div>
                <div className="h-12 bg-ma-green-800 rounded"></div>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">MA Red</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="h-12 bg-ma-red-400 rounded"></div>
                <div className="h-12 bg-ma-red-600 rounded"></div>
                <div className="h-12 bg-ma-red-800 rounded"></div>
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">MA Slate</h3>
              <div className="grid grid-cols-3 gap-2">
                <div className="h-12 bg-ma-slate-400 rounded"></div>
                <div className="h-12 bg-ma-slate-600 rounded"></div>
                <div className="h-12 bg-ma-slate-800 rounded"></div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={isConfirmOpen}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={() => {
          console.log('Confirmed!');
          setIsConfirmOpen(false);
        }}
        title="Confirmer l'action"
        description="Êtes-vous sûr de vouloir effectuer cette action ?"
        variant="destructive"
        confirmText="Supprimer"
        cancelText="Annuler"
      />
    </div>
  );
};