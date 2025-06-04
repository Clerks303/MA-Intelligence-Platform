import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { AlertWithIcon } from '../ui/alert';
import { Badge } from '../ui/badge';
import { 
  Upload,
  CloudUpload,
  FileText,
  CheckCircle,
  Loader2,
  X,
  Info
} from 'lucide-react';
import { cn } from '../../lib/utils';
import api from '../../services/api';

export function UploadDropzone({ onSuccess, className }) {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [updateExisting, setUpdateExisting] = useState(false);

  const { getRootProps, getInputProps, acceptedFiles, isDragActive, fileRejections } = useDropzone({
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handleUpload = async () => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', acceptedFiles[0]);
    formData.append('update_existing', updateExisting);

    try {
      const response = await api.post('/companies/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });
      setResult(response.data);
      if (onSuccess) onSuccess();
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = error?.response?.data?.detail || error?.message || 'Erreur lors de l\'upload';
      setResult({ error: errorMessage });
    } finally {
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setResult(null);
    acceptedFiles.length = 0;
  };

  const file = acceptedFiles[0];

  return (
    <Card className={cn("transition-all duration-300 hover:shadow-lg", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CloudUpload className="h-5 w-5 text-primary" />
          Import de données CSV
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Importez vos propres listes d'entreprises. Le fichier doit contenir les colonnes SIREN et nom d'entreprise.
        </p>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {!result ? (
          <>
            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200",
                "hover:border-primary hover:bg-muted/50",
                isDragActive ? "border-primary bg-primary/5" : "border-border",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              )}
            >
              <input {...getInputProps()} />
              <div className="space-y-3">
                <div className={cn(
                  "mx-auto w-16 h-16 rounded-full flex items-center justify-center transition-colors",
                  isDragActive ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
                )}>
                  <CloudUpload className="h-8 w-8" />
                </div>
                
                <div>
                  <p className="text-lg font-medium">
                    {isDragActive
                      ? 'Déposez le fichier ici...'
                      : 'Glissez-déposez votre fichier CSV'}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    ou cliquez pour sélectionner un fichier
                  </p>
                </div>
                
                <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
                  <FileText className="h-3 w-3" />
                  <span>Format CSV uniquement - Max 10MB</span>
                </div>
              </div>
            </div>

            {/* File Rejections */}
            {fileRejections.length > 0 && (
              <AlertWithIcon variant="destructive" title="Fichier rejeté">
                {fileRejections[0].errors.map(error => (
                  <div key={error.code}>{error.message}</div>
                ))}
              </AlertWithIcon>
            )}

            {/* Selected File */}
            {file && (
              <div className="border rounded-lg p-4 bg-muted/20">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded bg-primary/10 text-primary">
                      <FileText className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={resetUpload}
                    className="h-8 w-8 p-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}

            {/* Options */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="updateExisting"
                checked={updateExisting}
                onChange={(e) => setUpdateExisting(e.target.checked)}
                className="rounded border-border"
              />
              <label htmlFor="updateExisting" className="text-sm font-medium">
                Mettre à jour les entreprises existantes
              </label>
            </div>

            {/* Info */}
            <AlertWithIcon variant="info">
              <div className="text-sm">
                <p className="font-medium mb-1">Colonnes requises :</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>SIREN (obligatoire)</li>
                  <li>Nom entreprise (obligatoire)</li>
                  <li>Email, téléphone, adresse (optionnels)</li>
                </ul>
              </div>
            </AlertWithIcon>

            {/* Upload Button */}
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="w-full gap-2"
            >
              {uploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Import en cours...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  Importer le fichier
                </>
              )}
            </Button>
          </>
        ) : (
          /* Results */
          <div className="space-y-4">
            {result?.error ? (
              <AlertWithIcon variant="destructive" title="Erreur d'import">
                {result?.error || 'Erreur inconnue'}
              </AlertWithIcon>
            ) : (
              <AlertWithIcon variant="success" title="Import terminé avec succès !">
                <div className="space-y-3 mt-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center justify-between p-2 bg-background rounded">
                      <span>Total en base</span>
                      <Badge variant="secondary">{result?.total_rows || 0}</Badge>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-background rounded">
                      <span>Nouvelles ajoutées</span>
                      <Badge variant="success">{result?.new_companies || 0}</Badge>
                    </div>
                    {updateExisting && (
                      <div className="flex items-center justify-between p-2 bg-background rounded">
                        <span>Mises à jour</span>
                        <Badge variant="info">{result?.updated_companies || 0}</Badge>
                      </div>
                    )}
                    <div className="flex items-center justify-between p-2 bg-background rounded">
                      <span>Ignorées</span>
                      <Badge variant="outline">{result?.skipped_companies || 0}</Badge>
                    </div>
                  </div>
                  
                  {/* Section Entreprises */}
                  {result?.entreprises && (
                    <div className="mt-4 p-3 bg-muted/30 rounded-lg">
                      <h5 className="font-medium mb-2 text-sm">Table Entreprises</h5>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex items-center justify-between p-1">
                          <span>Total entreprises</span>
                          <Badge variant="secondary">{result?.entreprises?.total || 0}</Badge>
                        </div>
                        <div className="flex items-center justify-between p-1">
                          <span>Nouvelles</span>
                          <Badge variant="success">{result?.entreprises?.nouvelles || 0}</Badge>
                        </div>
                        <div className="flex items-center justify-between p-1">
                          <span>Mises à jour</span>
                          <Badge variant="info">{result?.entreprises?.mises_a_jour || 0}</Badge>
                        </div>
                        <div className="flex items-center justify-between p-1">
                          <span>Ignorées</span>
                          <Badge variant="outline">{result?.entreprises?.ignorees || 0}</Badge>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </AlertWithIcon>
            )}

            <Button
              onClick={resetUpload}
              variant="outline"
              className="w-full gap-2"
            >
              <Upload className="h-4 w-4" />
              Importer un autre fichier
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}