# Composants Monitoring Dashboard - US-004

Interface frontend complète pour le monitoring temps réel de la plateforme M&A Intelligence.

## 🏗️ Architecture

### MonitoringContext
Context provider React avec React Query pour la gestion d'état global du monitoring.

**Features:**
- Polling automatique toutes les 30 secondes
- Gestion des alertes avec acquittement/résolution
- Notifications en temps réel
- Cache intelligent avec React Query
- Actions async pour API monitoring

**Usage:**
```jsx
import { MonitoringProvider, useMonitoring } from './contexts/MonitoringContext';

function App() {
  return (
    <MonitoringProvider>
      <MonitoringDashboard />
    </MonitoringProvider>
  );
}

function Component() {
  const { 
    overview, 
    alerts, 
    loading, 
    acknowledgeAlert, 
    toggleRealTime 
  } = useMonitoring();
}
```

### Composants Dashboard

#### MonitoringDashboard
Composant principal orchestrant tous les widgets de monitoring.

**Props:** Aucune (utilise le context)

**Features:**
- Layout responsive Material-UI Grid
- Contrôles temps réel (toggle, refresh, paramètres)
- Indicateur santé système
- Notifications intégrées
- Menu paramètres avec vues multiples

#### SystemOverviewCard
Widget vue d'ensemble avec KPIs système principaux.

**Props:**
- `data`: Données overview système
- `loading`: État de chargement
- `onRefresh`: Callback refresh

**KPIs Affichés:**
- Temps de réponse API moyenne
- Requests par heure
- Taux d'erreur
- Utilisateurs actifs
- Usage CPU/Mémoire
- Métriques business (companies scrapées, scores M&A)

#### AlertsOverview
Interface de gestion des alertes avec actions.

**Features:**
- Liste alertes actives avec filtres
- Icônes par sévérité (Emergency, Critical, Warning, Info)
- Actions acquittement/résolution avec dialog
- Filtrage par sévérité et statut
- Déduplication et groupement

**Actions disponibles:**
```jsx
const { acknowledgeAlert, resolveAlert } = useMonitoring();

// Acquitter une alerte
await acknowledgeAlert(alertId, "Investigating issue");

// Résoudre une alerte
await resolveAlert(alertId, "Fixed by restarting service");
```

#### MetricsChart
Charts temps réel avec Recharts, multiple types et périodes.

**Props:**
- `title`: Titre du chart
- `metric`: Type de métrique à afficher
- `timeWindow`: Fenêtre temporelle ("5m", "15m", "1h", etc.)
- `chartType`: Type de chart ("line", "area", "bar")
- `height`: Hauteur du chart (défaut: 300px)

**Types de métriques supportés:**
- `api_response_time`: Temps de réponse API
- `system_resources`: Usage CPU/Mémoire
- `business_activity`: Activité business

**Types de charts:**
- LineChart: Évolution temporelle
- AreaChart: Zones avec remplissage
- BarChart: Barres pour données discrètes

#### HealthStatus
Widget santé des services avec diagnostics détaillés.

**Features:**
- Vue d'ensemble santé globale avec %
- Liste services avec statuts (Healthy, Degraded, Unhealthy)
- Mode simple/détaillé avec diagnostics
- Icônes par type de service (Database, Cache, API, External)
- Métriques par service (temps réponse, connexions, etc.)

#### RealTimeIndicator
Indicateur visuel du statut temps réel avec animations.

**Props:**
- `enabled`: État temps réel activé/désactivé
- `lastUpdated`: Timestamp dernière mise à jour
- `error`: Erreur de connexion

**États supportés:**
- `connected`: Temps réel actif (vert, animé)
- `warning`: Connexion lente (orange)
- `stale`: Connexion perdue (rouge)
- `disabled`: Mode manuel (gris)
- `error`: Erreur de connexion (rouge)

## 🎨 Design & Responsive

### Material-UI Integration
Tous les composants utilisent Material-UI v5 avec:
- Grid system responsive (xs, sm, md, lg, xl)
- Theme provider pour cohérence visuelle
- Icons Material (@mui/icons-material)
- Composants Cards, Chips, Progress, Tooltips

### Breakpoints
```jsx
<Grid container spacing={3}>
  <Grid item xs={12} lg={8}>        {/* Large sur desktop */}
    <SystemOverviewCard />
  </Grid>
  <Grid item xs={12} lg={4}>        {/* Sidebar sur desktop */}
    <AlertsOverview />
  </Grid>
  <Grid item xs={12} md={6}>        {/* 2 colonnes sur tablet+ */}
    <MetricsChart />
  </Grid>
</Grid>
```

### Thème Sombre
Support automatique du thème sombre via ThemeContext existant.

## 📊 Intégration API

### Endpoints Utilisés
- `GET /monitoring/overview`: Vue d'ensemble système
- `GET /monitoring/alerts`: Alertes avec filtres
- `GET /monitoring/metrics`: Métriques détaillées
- `GET /monitoring/health`: Santé services
- `POST /monitoring/alerts/{id}/acknowledge`: Acquitter alerte
- `POST /monitoring/alerts/{id}/resolve`: Résoudre alerte

### React Query Configuration
```js
const overviewQuery = useQuery(
  'monitoring-overview',
  monitoringApi.getOverview,
  {
    enabled: realTimeEnabled,
    refetchInterval: 30000,        // 30 secondes
    keepPreviousData: true,
    onSuccess: (data) => {
      dispatch({ type: 'SET_OVERVIEW', payload: data });
    }
  }
);
```

## 🧪 Tests

### Tests Unitaires
- `MonitoringContext.test.js`: Tests du context provider
- `MonitoringDashboard.test.js`: Tests du dashboard principal

### Exécution
```bash
npm test -- --testPathPattern=monitoring
```

### Validation Complète
```bash
node scripts/validate_us004_implementation.js --comprehensive
```

## 🚀 Utilisation

### 1. Import dans App.js
```jsx
import Monitoring from './pages/Monitoring';

// Ajouter route
<Route path="monitoring" element={<Monitoring />} />
```

### 2. Navigation Layout.js
```jsx
const menuItems = [
  { text: 'Monitoring', icon: Activity, path: '/monitoring' }
];
```

### 3. Composant de base
```jsx
import { MonitoringProvider } from './contexts/MonitoringContext';
import { MonitoringDashboard } from './components/monitoring';

function MonitoringPage() {
  return (
    <MonitoringProvider>
      <Container maxWidth={false}>
        <MonitoringDashboard />
      </Container>
    </MonitoringProvider>
  );
}
```

## 🔧 Configuration

### Intervalles de Polling
```js
const config = {
  refreshInterval: 30000,        // 30s - données overview
  metricsInterval: 60000,        // 1min - métriques détaillées
  notificationTimeout: 5000      // 5s - auto-remove notifications
};
```

### Filtres Alertes par Défaut
```js
const defaultFilters = {
  severity: 'all',               // all, critical, warning, info
  status: 'active',              // all, active, acknowledged
  category: 'all'                // all, system, application, business
};
```

## 📱 Features Avancées

### Notifications Browser
Demande automatique de permission et notifications critiques.

### Gestion Erreurs
Retry automatique et fallback graceful en cas d'erreur API.

### Performance
- Debounce des actions utilisateur
- Memoization des calculs coûteux
- Virtualisation pour grandes listes
- Lazy loading des composants

### Accessibilité
- Labels ARIA appropriés
- Navigation clavier
- Contraste couleurs respecté
- Tooltips descriptifs

## 🎯 Prochaines Améliorations

1. **WebSocket Real-time**: Remplacer polling par WebSocket
2. **Export Données**: Export PDF/Excel des métriques
3. **Alertes Personnalisées**: Interface création règles custom
4. **Dashboard Personnalisable**: Widgets déplaçables
5. **Historique Détaillé**: Charts historiques longue durée
6. **Annotations**: Marqueurs d'événements sur charts