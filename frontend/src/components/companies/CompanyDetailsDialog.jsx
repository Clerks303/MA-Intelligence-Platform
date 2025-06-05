import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/tabs';
import { StatusBadge } from './StatusBadge';
import { 
  Building2,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Euro,
  Users,
  TrendingUp,
  Edit,
  ExternalLink,
  Star
} from 'lucide-react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import api from '../../services/api';
import { cn } from '../../lib/utils';

export function CompanyDetailsDialog({ open, onClose, siren }) {
  const [activeTab, setActiveTab] = useState('general');
  
  const { data: company, isLoading } = useQuery({
    queryKey: ['company', siren],
    queryFn: () => api.get(`/companies/${siren}`).then(res => res.data),
    enabled: !!siren && open
  });

  const formatCurrency = (amount) => {
    if (!amount) return 'N/A';
    return new Intl.NumberFormat('fr-FR', { 
      style: 'currency', 
      currency: 'EUR' 
    }).format(amount);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return format(new Date(dateString), 'dd MMMM yyyy', { locale: fr });
    } catch {
      return 'N/A';
    }
  };

  if (isLoading) {
    return (
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent>
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center space-x-2">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="text-muted-foreground">Chargement des détails...</span>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  if (!company) {
    return (
      <Dialog open={open} onOpenChange={onClose}>
        <DialogContent>
          <DialogHeader onClose={onClose}>
            <DialogTitle>Erreur</DialogTitle>
          </DialogHeader>
          <DialogContent>
            <p className="text-muted-foreground">Impossible de charger les détails de l'entreprise.</p>
          </DialogContent>
          <DialogFooter>
            <Button onClick={onClose}>Fermer</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
        <DialogHeader onClose={onClose}>
          <div className="flex items-start justify-between">
            <div>
              <DialogTitle className="text-xl flex items-center gap-3">
                <Building2 className="h-6 w-6 text-primary" />
                {company.nom_entreprise}
              </DialogTitle>
              <div className="flex items-center gap-2 mt-2">
                <StatusBadge status={company.statut} />
                {company?.score_prospection && (
                  <Badge variant="outline" className="gap-1">
                    <Star className="h-3 w-3" />
                    Score: {Math.round(company?.score_prospection || 0)}%
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </DialogHeader>

        <DialogContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="general">Général</TabsTrigger>
              <TabsTrigger value="financial">Financier</TabsTrigger>
              <TabsTrigger value="management">Dirigeants</TabsTrigger>
              <TabsTrigger value="history">Historique</TabsTrigger>
            </TabsList>

            <TabsContent value="general" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-semibold text-lg flex items-center gap-2">
                    <Building2 className="h-5 w-5" />
                    Informations générales
                  </h4>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">SIREN</label>
                      <p className="font-mono text-sm">{company?.siren || 'N/A'}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">SIRET Siège</label>
                      <p className="font-mono text-sm">{company.siret_siege || 'N/A'}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Forme juridique</label>
                      <p>{company.forme_juridique || 'N/A'}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Code NAF</label>
                      <p>{company.code_naf ? `${company.code_naf} - ${company.libelle_code_naf}` : 'N/A'}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Date de création</label>
                      <p className="flex items-center gap-2">
                        <Calendar className="h-4 w-4" />
                        {formatDate(company.date_creation)}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-lg flex items-center gap-2">
                    <MapPin className="h-5 w-5" />
                    Contact
                  </h4>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Adresse</label>
                      <p className="text-sm">{company.adresse || 'N/A'}</p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Email</label>
                      <p className="flex items-center gap-2">
                        <Mail className="h-4 w-4" />
                        {company?.email ? (
                          <a 
                            href={`mailto:${company?.email}`}
                            className="text-primary hover:underline"
                          >
                            {company?.email}
                          </a>
                        ) : 'N/A'}
                      </p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">Téléphone</label>
                      <p className="flex items-center gap-2">
                        <Phone className="h-4 w-4" />
                        {company?.telephone ? (
                          <a 
                            href={`tel:${company?.telephone}`}
                            className="text-primary hover:underline"
                          >
                            {company?.telephone}
                          </a>
                        ) : 'N/A'}
                      </p>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-muted-foreground">N° TVA</label>
                      <p className="font-mono text-sm">{company.numero_tva || 'N/A'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {company?.score_prospection && (
                <div className="border rounded-lg p-4 bg-muted/20">
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Score de prospection
                  </h4>
                  <div className="flex items-center space-x-4">
                    <div className="flex-1 bg-muted rounded-full h-3">
                      <div 
                        className="bg-primary h-3 rounded-full transition-all duration-500"
                        style={{ width: `${company?.score_prospection || 0}%` }}
                      />
                    </div>
                    <span className="font-semibold text-lg">
                      {Math.round(company?.score_prospection || 0)}%
                    </span>
                  </div>
                  {company.score_details && (
                    <p className="text-sm text-muted-foreground mt-2">
                      Basé sur les critères: activité, taille, croissance
                    </p>
                  )}
                </div>
              )}
            </TabsContent>

            <TabsContent value="financial" className="space-y-6">
              <h4 className="font-semibold text-lg flex items-center gap-2">
                <Euro className="h-5 w-5" />
                Données financières
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="bg-blue-50 dark:bg-blue-950 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Chiffre d'affaires</p>
                      <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                        {formatCurrency(company.chiffre_affaires)}
                      </p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-blue-500" />
                  </div>
                </div>

                <div className="bg-green-50 dark:bg-green-950 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-600 dark:text-green-400">Résultat</p>
                      <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                        {formatCurrency(company.resultat)}
                      </p>
                    </div>
                    <Euro className="h-8 w-8 text-green-500" />
                  </div>
                </div>

                <div className="bg-purple-50 dark:bg-purple-950 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-purple-600 dark:text-purple-400">Effectif</p>
                      <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                        {company.effectif || 'N/A'}
                      </p>
                    </div>
                    <Users className="h-8 w-8 text-purple-500" />
                  </div>
                </div>

                <div className="bg-orange-50 dark:bg-orange-950 rounded-lg p-4 md:col-span-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-orange-600 dark:text-orange-400">Capital social</p>
                      <p className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                        {formatCurrency(company.capital_social)}
                      </p>
                    </div>
                    <Building2 className="h-8 w-8 text-orange-500" />
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="management" className="space-y-6">
              <h4 className="font-semibold text-lg flex items-center gap-2">
                <Users className="h-5 w-5" />
                Équipe dirigeante
              </h4>
              
              {company.dirigeant_principal && (
                <div className="border rounded-lg p-4">
                  <h5 className="font-semibold mb-2">Dirigeant principal</h5>
                  <p className="text-lg">{company.dirigeant_principal}</p>
                </div>
              )}

              {company.dirigeants_json && (
                <div className="space-y-3">
                  <h5 className="font-semibold">Autres dirigeants</h5>
                  <div className="space-y-2">
                    {JSON.parse(company.dirigeants_json).map((dirigeant, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <p className="font-medium">{dirigeant.nom_complet}</p>
                          <p className="text-sm text-muted-foreground">{dirigeant.qualite}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!company.dirigeant_principal && !company.dirigeants_json && (
                <p className="text-muted-foreground text-center py-8">
                  Aucune information sur les dirigeants disponible.
                </p>
              )}
            </TabsContent>

            <TabsContent value="history" className="space-y-6">
              <h4 className="font-semibold text-lg flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Historique des actions
              </h4>
              
              {company.activity_logs && company.activity_logs.length > 0 ? (
                <div className="space-y-3">
                  {company.activity_logs.map((log, index) => (
                    <div key={index} className="flex items-start space-x-3 p-3 border rounded-lg">
                      <div className="w-2 h-2 bg-primary rounded-full mt-2" />
                      <div className="flex-1">
                        <p className="font-medium">{log.action}</p>
                        <p className="text-sm text-muted-foreground">
                          {formatDate(log.created_at)}
                        </p>
                        {log.details && (
                          <p className="text-sm mt-1">{log.details}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  Aucun historique d'activité disponible.
                </p>
              )}
            </TabsContent>
          </Tabs>
        </DialogContent>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Fermer
          </Button>
          <Button className="gap-2">
            <Edit className="h-4 w-4" />
            Modifier
          </Button>
          {company.lien_pappers && (
            <Button variant="outline" className="gap-2" asChild>
              <a href={company.lien_pappers} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
                Voir sur Pappers
              </a>
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}