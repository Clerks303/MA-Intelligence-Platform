import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { AlertWithIcon } from '../ui/alert';
import { 
  Building2, 
  Loader2, 
  Save,
  MapPin,
  Mail,
  Phone,
  Euro,
  Users
} from 'lucide-react';
import api from '../../services/api';

export function AddCompanyDialog({ open, onClose }) {
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState({
    siren: '',
    nom_entreprise: '',
    adresse: '',
    ville: '',
    code_postal: '',
    dirigeant_principal: '',
    email: '',
    telephone: '',
    chiffre_affaires: '',
    effectif: '',
    description: ''
  });
  const [errors, setErrors] = useState({});

  const createMutation = useMutation({
    mutationFn: (companyData) => api.post('/companies/', companyData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['companies'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      onClose();
      resetForm();
    },
    onError: (error) => {
      console.error('Error creating company:', error);
      
      if (error.response?.status === 400) {
        setErrors({ general: 'Ce SIREN existe déjà dans la base de données.' });
      } else if (error.response?.status === 422) {
        // Handle validation errors (422)
        const detail = error.response?.data?.detail;
        if (Array.isArray(detail)) {
          // Multiple field errors
          const fieldErrors = {};
          const generalErrors = [];
          
          detail.forEach(err => {
            if (err.loc && err.loc.length > 0) {
              const field = err.loc[err.loc.length - 1];
              fieldErrors[field] = err.msg;
            } else {
              generalErrors.push(err.msg);
            }
          });
          
          setErrors({
            ...fieldErrors,
            general: generalErrors.length > 0 ? generalErrors.join(', ') : 'Erreurs de validation détectées.'
          });
        } else if (typeof detail === 'string') {
          setErrors({ general: detail });
        } else {
          setErrors({ general: 'Erreurs de validation. Vérifiez vos données.' });
        }
      } else {
        // Handle other errors
        const detail = error.response?.data?.detail;
        const message = typeof detail === 'string' ? detail : 'Erreur lors de la création.';
        setErrors({ general: message });
      }
    }
  });

  const resetForm = () => {
    setFormData({
      siren: '',
      nom_entreprise: '',
      adresse: '',
      ville: '',
      code_postal: '',
      dirigeant_principal: '',
      email: '',
      telephone: '',
      chiffre_affaires: '',
      effectif: '',
      description: ''
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.siren.trim()) {
      newErrors.siren = 'SIREN requis';
    } else if (!/^\d{9}$/.test(formData.siren)) {
      newErrors.siren = 'SIREN doit contenir exactement 9 chiffres';
    }
    
    if (!formData.nom_entreprise.trim()) {
      newErrors.nom_entreprise = 'Nom d\'entreprise requis';
    }

    if (formData.email && !/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email invalide';
    }

    if (formData.chiffre_affaires && isNaN(Number(formData.chiffre_affaires))) {
      newErrors.chiffre_affaires = 'Montant invalide';
    }

    if (formData.effectif && isNaN(Number(formData.effectif))) {
      newErrors.effectif = 'Nombre invalide';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    const submitData = {
      ...formData,
      chiffre_affaires: formData.chiffre_affaires && formData.chiffre_affaires.trim() ? Number(formData.chiffre_affaires) : null,
      effectif: formData.effectif && formData.effectif.trim() ? Number(formData.effectif) : null,
      statut: 'prospect'
    };

    // Remove empty string fields to avoid validation errors
    Object.keys(submitData).forEach(key => {
      if (submitData[key] === '') {
        submitData[key] = null;
      }
    });

    createMutation.mutate(submitData);
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-primary" />
            Ajouter une entreprise
          </DialogTitle>
          <DialogDescription>
            Saisissez les informations de la nouvelle entreprise à ajouter à votre base de données.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {errors.general && (
            <AlertWithIcon variant="destructive">
              {errors.general}
            </AlertWithIcon>
          )}

          {/* Informations principales */}
          <div className="space-y-4">
            <h3 className="font-medium text-foreground">Informations principales</h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="siren">SIREN *</Label>
                <Input
                  id="siren"
                  value={formData.siren}
                  onChange={(e) => handleChange('siren', e.target.value)}
                  placeholder="123456789"
                  maxLength={9}
                  className={errors.siren ? 'border-destructive' : ''}
                />
                {errors.siren && (
                  <p className="text-sm text-destructive">{errors.siren}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="nom_entreprise">Nom de l'entreprise *</Label>
                <Input
                  id="nom_entreprise"
                  value={formData.nom_entreprise}
                  onChange={(e) => handleChange('nom_entreprise', e.target.value)}
                  placeholder="Nom de l'entreprise"
                  className={errors.nom_entreprise ? 'border-destructive' : ''}
                />
                {errors.nom_entreprise && (
                  <p className="text-sm text-destructive">{errors.nom_entreprise}</p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="dirigeant_principal">Dirigeant principal</Label>
              <Input
                id="dirigeant_principal"
                value={formData.dirigeant_principal}
                onChange={(e) => handleChange('dirigeant_principal', e.target.value)}
                placeholder="Nom du dirigeant"
              />
            </div>
          </div>

          {/* Adresse */}
          <div className="space-y-4">
            <h3 className="font-medium text-foreground flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Adresse
            </h3>
            
            <div className="space-y-2">
              <Label htmlFor="adresse">Adresse complète</Label>
              <Input
                id="adresse"
                value={formData.adresse}
                onChange={(e) => handleChange('adresse', e.target.value)}
                placeholder="123 Rue de la Paix"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="ville">Ville</Label>
                <Input
                  id="ville"
                  value={formData.ville}
                  onChange={(e) => handleChange('ville', e.target.value)}
                  placeholder="Paris"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="code_postal">Code postal</Label>
                <Input
                  id="code_postal"
                  value={formData.code_postal}
                  onChange={(e) => handleChange('code_postal', e.target.value)}
                  placeholder="75001"
                  maxLength={5}
                />
              </div>
            </div>
          </div>

          {/* Contact */}
          <div className="space-y-4">
            <h3 className="font-medium text-foreground flex items-center gap-2">
              <Mail className="h-4 w-4" />
              Contact
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => handleChange('email', e.target.value)}
                  placeholder="contact@entreprise.com"
                  className={errors.email ? 'border-destructive' : ''}
                />
                {errors.email && (
                  <p className="text-sm text-destructive">{errors.email}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="telephone">Téléphone</Label>
                <Input
                  id="telephone"
                  value={formData.telephone}
                  onChange={(e) => handleChange('telephone', e.target.value)}
                  placeholder="01 23 45 67 89"
                />
              </div>
            </div>
          </div>

          {/* Informations financières */}
          <div className="space-y-4">
            <h3 className="font-medium text-foreground flex items-center gap-2">
              <Euro className="h-4 w-4" />
              Informations financières
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="chiffre_affaires">Chiffre d'affaires (€)</Label>
                <Input
                  id="chiffre_affaires"
                  type="number"
                  value={formData.chiffre_affaires}
                  onChange={(e) => handleChange('chiffre_affaires', e.target.value)}
                  placeholder="1000000"
                  className={errors.chiffre_affaires ? 'border-destructive' : ''}
                />
                {errors.chiffre_affaires && (
                  <p className="text-sm text-destructive">{errors.chiffre_affaires}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="effectif">Effectif</Label>
                <Input
                  id="effectif"
                  type="number"
                  value={formData.effectif}
                  onChange={(e) => handleChange('effectif', e.target.value)}
                  placeholder="10"
                  className={errors.effectif ? 'border-destructive' : ''}
                />
                {errors.effectif && (
                  <p className="text-sm text-destructive">{errors.effectif}</p>
                )}
              </div>
            </div>
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="description">Description / Notes</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              placeholder="Informations complémentaires..."
              rows={3}
            />
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                onClose();
                resetForm();
              }}
              disabled={createMutation.isLoading}
            >
              Annuler
            </Button>
            <Button
              type="submit"
              disabled={createMutation.isLoading}
              className="gap-2"
            >
              {createMutation.isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Création...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4" />
                  Créer l'entreprise
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}