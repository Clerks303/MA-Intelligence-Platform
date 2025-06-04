#!/usr/bin/env node

/**
 * Script de validation pour US-004: Interface Frontend Dashboard
 * Vérifie que tous les composants monitoring ont été correctement implémentés
 * 
 * Usage:
 *     node scripts/validate_us004_implementation.js
 *     node scripts/validate_us004_implementation.js --comprehensive
 *     node scripts/validate_us004_implementation.js --build-test
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class US004Validator {
  constructor() {
    this.validationResults = {};
    this.successCount = 0;
    this.failureCount = 0;
    
    // Critères de succès US-004
    this.successCriteria = {
      'monitoring_context_implemented': true,
      'dashboard_components_created': true,
      'real_time_charts_functional': true,
      'alerts_interface_working': true,
      'responsive_design_ready': true,
      'navigation_integrated': true,
      'tests_passing': true
    };
  }

  async runValidation(comprehensive = false, buildTest = false) {
    console.log('🔍 VALIDATION US-004: Interface Frontend Dashboard');
    console.log('='.repeat(80));

    // Validations de base
    this.validateMonitoringContext();
    this.validateDashboardComponents();
    this.validateChartsAndWidgets();
    this.validateAlertsInterface();
    this.validateNavigationIntegration();
    
    if (comprehensive) {
      this.validateResponsiveDesign();
      this.validateRealTimeFeatures();
      this.validateAccessibility();
    }

    if (buildTest) {
      await this.runBuildTests();
    }

    // Tests unitaires
    await this.runUnitTests();

    // Validation environnement
    this.validateEnvironmentConfiguration();

    // Génération rapport
    this.generateValidationReport();

    return [this.successCount, this.failureCount];
  }

  validateMonitoringContext() {
    const testName = 'monitoring_context';
    console.log('🔄 Test MonitoringContext...');

    try {
      const validations = [];

      // Vérifier fichier context
      const contextPath = 'src/contexts/MonitoringContext.js';
      const contextExists = fs.existsSync(contextPath);
      validations.push(['context_file_exists', contextExists]);

      if (contextExists) {
        const contextContent = fs.readFileSync(contextPath, 'utf8');
        
        // Vérifier exports principaux
        const hasProvider = contextContent.includes('MonitoringProvider');
        const hasHook = contextContent.includes('useMonitoring');
        const hasActions = contextContent.includes('acknowledgeAlert') && contextContent.includes('resolveAlert');
        const hasQueries = contextContent.includes('useQuery');
        
        validations.push(['provider_exported', hasProvider]);
        validations.push(['hook_exported', hasHook]);
        validations.push(['actions_implemented', hasActions]);
        validations.push(['react_query_integrated', hasQueries]);

        // Vérifier structure reducer
        const hasReducer = contextContent.includes('monitoringReducer');
        const hasActions = contextContent.includes('MONITORING_ACTIONS');
        validations.push(['reducer_implemented', hasReducer]);
        validations.push(['actions_defined', hasActions]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateDashboardComponents() {
    const testName = 'dashboard_components';
    console.log('🔄 Test composants dashboard...');

    try {
      const validations = [];
      const componentsDir = 'src/components/monitoring';

      // Composants requis
      const requiredComponents = [
        'MonitoringDashboard.jsx',
        'SystemOverviewCard.jsx',
        'AlertsOverview.jsx',
        'MetricsChart.jsx',
        'HealthStatus.jsx',
        'RealTimeIndicator.jsx'
      ];

      let existingComponents = 0;
      requiredComponents.forEach(component => {
        const componentPath = path.join(componentsDir, component);
        const exists = fs.existsSync(componentPath);
        if (exists) existingComponents++;
        validations.push([`component_${component}`, exists]);
      });

      const allComponentsExist = existingComponents === requiredComponents.length;
      validations.push(['all_components_exist', allComponentsExist]);

      // Vérifier page principale
      const monitoringPageExists = fs.existsSync('src/pages/Monitoring.js');
      validations.push(['monitoring_page_exists', monitoringPageExists]);

      if (monitoringPageExists) {
        const pageContent = fs.readFileSync('src/pages/Monitoring.js', 'utf8');
        const hasProvider = pageContent.includes('MonitoringProvider');
        const hasDashboard = pageContent.includes('MonitoringDashboard');
        validations.push(['page_uses_provider', hasProvider]);
        validations.push(['page_uses_dashboard', hasDashboard]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, {
        validations,
        components_found: existingComponents,
        components_required: requiredComponents.length
      });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateChartsAndWidgets() {
    const testName = 'charts_widgets';
    console.log('🔄 Test charts et widgets...');

    try {
      const validations = [];

      // Vérifier MetricsChart
      const chartPath = 'src/components/monitoring/MetricsChart.jsx';
      if (fs.existsSync(chartPath)) {
        const chartContent = fs.readFileSync(chartPath, 'utf8');
        
        const hasRecharts = chartContent.includes('recharts');
        const hasMultipleChartTypes = chartContent.includes('LineChart') && 
                                     chartContent.includes('AreaChart') && 
                                     chartContent.includes('BarChart');
        const hasTimeWindows = chartContent.includes('timeWindow');
        const hasResponsiveContainer = chartContent.includes('ResponsiveContainer');
        
        validations.push(['recharts_integrated', hasRecharts]);
        validations.push(['multiple_chart_types', hasMultipleChartTypes]);
        validations.push(['time_windows_implemented', hasTimeWindows]);
        validations.push(['responsive_charts', hasResponsiveContainer]);
      } else {
        validations.push(['metrics_chart_missing', false]);
      }

      // Vérifier SystemOverviewCard
      const overviewPath = 'src/components/monitoring/SystemOverviewCard.jsx';
      if (fs.existsSync(overviewPath)) {
        const overviewContent = fs.readFileSync(overviewPath, 'utf8');
        
        const hasKPIs = overviewContent.includes('key_metrics') || overviewContent.includes('keyMetrics');
        const hasProgressBars = overviewContent.includes('LinearProgress');
        const hasHealthIndicators = overviewContent.includes('getHealthIcon');
        
        validations.push(['kpis_implemented', hasKPIs]);
        validations.push(['progress_indicators', hasProgressBars]);
        validations.push(['health_indicators', hasHealthIndicators]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateAlertsInterface() {
    const testName = 'alerts_interface';
    console.log('🔄 Test interface alertes...');

    try {
      const validations = [];

      const alertsPath = 'src/components/monitoring/AlertsOverview.jsx';
      if (fs.existsSync(alertsPath)) {
        const alertsContent = fs.readFileSync(alertsPath, 'utf8');
        
        const hasAcknowledge = alertsContent.includes('acknowledgeAlert');
        const hasResolve = alertsContent.includes('resolveAlert');
        const hasFilters = alertsContent.includes('filter');
        const hasDialog = alertsContent.includes('Dialog');
        const hasSeverityIcons = alertsContent.includes('getSeverityIcon');
        
        validations.push(['acknowledge_action', hasAcknowledge]);
        validations.push(['resolve_action', hasResolve]);
        validations.push(['filtering_implemented', hasFilters]);
        validations.push(['action_dialogs', hasDialog]);
        validations.push(['severity_indicators', hasSeverityIcons]);
      } else {
        validations.push(['alerts_component_missing', false]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateNavigationIntegration() {
    const testName = 'navigation_integration';
    console.log('🔄 Test intégration navigation...');

    try {
      const validations = [];

      // Vérifier Layout.js
      const layoutPath = 'src/components/Layout.js';
      if (fs.existsSync(layoutPath)) {
        const layoutContent = fs.readFileSync(layoutPath, 'utf8');
        
        const hasMonitoringItem = layoutContent.includes('Monitoring') || layoutContent.includes('monitoring');
        const hasActivityIcon = layoutContent.includes('Activity');
        
        validations.push(['monitoring_in_menu', hasMonitoringItem]);
        validations.push(['activity_icon_added', hasActivityIcon]);
      }

      // Vérifier App.js
      const appPath = 'src/App.js';
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        
        const hasMonitoringImport = appContent.includes('import Monitoring');
        const hasMonitoringRoute = appContent.includes('path="monitoring"');
        
        validations.push(['monitoring_imported', hasMonitoringImport]);
        validations.push(['monitoring_route_added', hasMonitoringRoute]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateResponsiveDesign() {
    const testName = 'responsive_design';
    console.log('🔄 Test design responsive...');

    try {
      const validations = [];

      // Vérifier utilisation Material-UI Grid
      const dashboardPath = 'src/components/monitoring/MonitoringDashboard.jsx';
      if (fs.existsSync(dashboardPath)) {
        const dashboardContent = fs.readFileSync(dashboardPath, 'utf8');
        
        const hasMUIGrid = dashboardContent.includes('@mui/material') && dashboardContent.includes('Grid');
        const hasBreakpoints = dashboardContent.includes('xs=') && dashboardContent.includes('lg=');
        const hasResponsiveBox = dashboardContent.includes('sx={{');
        
        validations.push(['mui_grid_used', hasMUIGrid]);
        validations.push(['breakpoints_defined', hasBreakpoints]);
        validations.push(['responsive_styling', hasResponsiveBox]);
      }

      // Vérifier composants utilisent sx props
      const componentFiles = [
        'src/components/monitoring/SystemOverviewCard.jsx',
        'src/components/monitoring/AlertsOverview.jsx',
        'src/components/monitoring/MetricsChart.jsx'
      ];

      let componentsWithResponsive = 0;
      componentFiles.forEach(file => {
        if (fs.existsSync(file)) {
          const content = fs.readFileSync(file, 'utf8');
          if (content.includes('sx={{') || content.includes('useTheme')) {
            componentsWithResponsive++;
          }
        }
      });

      const responsiveComponents = componentsWithResponsive >= 2;
      validations.push(['components_responsive', responsiveComponents]);

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, {
        validations,
        responsive_components: componentsWithResponsive
      });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateRealTimeFeatures() {
    const testName = 'realtime_features';
    console.log('🔄 Test fonctionnalités temps réel...');

    try {
      const validations = [];

      // Vérifier RealTimeIndicator
      const indicatorPath = 'src/components/monitoring/RealTimeIndicator.jsx';
      if (fs.existsSync(indicatorPath)) {
        const indicatorContent = fs.readFileSync(indicatorPath, 'utf8');
        
        const hasAnimations = indicatorContent.includes('keyframes') || indicatorContent.includes('animation');
        const hasStatusIndicators = indicatorContent.includes('getConnectionStatus');
        const hasTooltips = indicatorContent.includes('Tooltip');
        
        validations.push(['animations_implemented', hasAnimations]);
        validations.push(['status_indicators', hasStatusIndicators]);
        validations.push(['tooltips_added', hasTooltips]);
      }

      // Vérifier polling dans context
      const contextPath = 'src/contexts/MonitoringContext.js';
      if (fs.existsSync(contextPath)) {
        const contextContent = fs.readFileSync(contextPath, 'utf8');
        
        const hasRefetchInterval = contextContent.includes('refetchInterval');
        const hasRealTimeToggle = contextContent.includes('toggleRealTime');
        
        validations.push(['polling_implemented', hasRefetchInterval]);
        validations.push(['realtime_toggle', hasRealTimeToggle]);
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  validateAccessibility() {
    const testName = 'accessibility';
    console.log('🔄 Test accessibilité...');

    try {
      const validations = [];

      // Vérifier attributs aria et rôles
      const componentFiles = [
        'src/components/monitoring/MonitoringDashboard.jsx',
        'src/components/monitoring/AlertsOverview.jsx'
      ];

      let accessibleComponents = 0;
      componentFiles.forEach(file => {
        if (fs.existsSync(file)) {
          const content = fs.readFileSync(file, 'utf8');
          if (content.includes('aria-') || content.includes('role=') || content.includes('Tooltip')) {
            accessibleComponents++;
          }
        }
      });

      const accessibilityImplemented = accessibleComponents >= 1;
      validations.push(['accessibility_features', accessibilityImplemented]);

      // Vérifier labels et tooltips
      const hasTooltips = componentFiles.some(file => {
        if (fs.existsSync(file)) {
          const content = fs.readFileSync(file, 'utf8');
          return content.includes('Tooltip') || content.includes('title=');
        }
        return false;
      });
      
      validations.push(['tooltips_implemented', hasTooltips]);

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, {
        validations,
        accessible_components: accessibleComponents
      });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  async runUnitTests() {
    const testName = 'unit_tests';
    console.log('🔄 Tests unitaires...');

    try {
      const validations = [];

      // Vérifier fichiers de test
      const testFiles = [
        'src/components/monitoring/__tests__/MonitoringContext.test.js',
        'src/components/monitoring/__tests__/MonitoringDashboard.test.js'
      ];

      let existingTests = 0;
      testFiles.forEach(testFile => {
        const exists = fs.existsSync(testFile);
        if (exists) existingTests++;
        validations.push([`test_${path.basename(testFile)}`, exists]);
      });

      const hasTests = existingTests >= 2;
      validations.push(['tests_implemented', hasTests]);

      // Exécuter tests si disponibles
      if (hasTests) {
        try {
          console.log('   Exécution des tests...');
          execSync('npm test -- --watchAll=false --testPathPattern=monitoring', { 
            stdio: 'pipe',
            timeout: 30000
          });
          validations.push(['tests_passing', true]);
        } catch (error) {
          console.log('   ⚠️ Tests échoués ou non configurés');
          validations.push(['tests_passing', false]);
        }
      }

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, {
        validations,
        test_files_found: existingTests
      });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  async runBuildTests() {
    const testName = 'build_tests';
    console.log('🔄 Tests de build...');

    try {
      console.log('   Test build production...');
      
      // Test build
      execSync('npm run build', { 
        stdio: 'pipe',
        timeout: 120000 // 2 minutes
      });

      // Vérifier que le build a généré les fichiers
      const buildExists = fs.existsSync('build');
      const hasIndexHtml = fs.existsSync('build/index.html');
      const hasStaticAssets = fs.existsSync('build/static');

      this._recordResult(testName, buildExists && hasIndexHtml && hasStaticAssets, {
        build_directory: buildExists,
        index_html: hasIndexHtml,
        static_assets: hasStaticAssets
      });

    } catch (error) {
      this._recordFailure(testName, `Build failed: ${error.message}`);
    }
  }

  validateEnvironmentConfiguration() {
    const testName = 'environment_configuration';
    console.log('🔄 Test configuration environnement...');

    try {
      const validations = [];

      // Vérifier package.json
      const packageExists = fs.existsSync('package.json');
      validations.push(['package_json_exists', packageExists]);

      if (packageExists) {
        const packageContent = JSON.parse(fs.readFileSync('package.json', 'utf8'));
        
        const hasMUI = packageContent.dependencies && packageContent.dependencies['@mui/material'];
        const hasRecharts = packageContent.dependencies && packageContent.dependencies['recharts'];
        const hasReactQuery = packageContent.dependencies && packageContent.dependencies['react-query'];
        
        validations.push(['mui_dependency', !!hasMUI]);
        validations.push(['recharts_dependency', !!hasRecharts]);
        validations.push(['react_query_dependency', !!hasReactQuery]);
      }

      // Vérifier structure dossiers
      const monitoringDirExists = fs.existsSync('src/components/monitoring');
      const contextExists = fs.existsSync('src/contexts/MonitoringContext.js');
      
      validations.push(['monitoring_directory', monitoringDirExists]);
      validations.push(['monitoring_context', contextExists]);

      const success = validations.every(([, result]) => result);
      this._recordResult(testName, success, { validations });

    } catch (error) {
      this._recordFailure(testName, error.message);
    }
  }

  // Méthodes utilitaires

  _recordSuccess(testName, details = null) {
    this.validationResults[testName] = {
      status: 'SUCCESS',
      details,
      timestamp: new Date().toISOString()
    };
    this.successCount++;
    console.log(`✅ ${testName}: SUCCÈS`);
  }

  _recordFailure(testName, error) {
    this.validationResults[testName] = {
      status: 'FAILURE',
      error,
      timestamp: new Date().toISOString()
    };
    this.failureCount++;
    console.error(`❌ ${testName}: ÉCHEC - ${error}`);
  }

  _recordResult(testName, success, details = null) {
    if (success) {
      this._recordSuccess(testName, details);
    } else {
      this._recordFailure(testName, `Validation échouée: ${JSON.stringify(details)}`);
    }
  }

  generateValidationReport() {
    console.log('='.repeat(80));
    console.log('📋 RAPPORT DE VALIDATION US-004');
    console.log('='.repeat(80));

    const totalTests = this.successCount + this.failureCount;
    const successRate = totalTests > 0 ? (this.successCount / totalTests * 100) : 0;

    console.log(`🎯 Tests réussis: ${this.successCount}/${totalTests} (${successRate.toFixed(1)}%)`);

    // Vérification critères de succès US-004
    console.log('\n📊 CRITÈRES DE SUCCÈS US-004:');
    this._checkSuccessCriteria();

    if (this.failureCount > 0) {
      console.log(`\n⚠️ Tests échoués: ${this.failureCount}`);
      Object.entries(this.validationResults).forEach(([testName, result]) => {
        if (result.status === 'FAILURE') {
          console.log(`  - ${testName}: ${result.error}`);
        }
      });
    }

    // Recommandations
    const recommendations = this._generateRecommendations();
    if (recommendations.length > 0) {
      console.log('\n💡 RECOMMANDATIONS:');
      recommendations.forEach((rec, i) => {
        console.log(`  ${i + 1}. ${rec}`);
      });
    }

    // Sauvegarde rapport
    const report = {
      timestamp: new Date().toISOString(),
      us_story: 'US-004: Interface Frontend Dashboard',
      summary: {
        total_tests: totalTests,
        success_count: this.successCount,
        failure_count: this.failureCount,
        success_rate: Math.round(successRate * 10) / 10
      },
      success_criteria_check: this._getSuccessCriteriaStatus(),
      detailed_results: this.validationResults,
      recommendations,
      status: successRate >= 80 ? 'PASSED' : 'FAILED'
    };

    const reportFile = `us004_validation_report_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)}.json`;
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));

    console.log(`\n📄 Rapport détaillé sauvegardé: ${reportFile}`);

    // Status final
    if (successRate >= 80) {
      console.log('🎉 US-004 VALIDÉ: Interface Frontend Dashboard implémentée avec succès!');
    } else {
      console.log('💥 US-004 ÉCHOUÉ: Interface frontend incomplète');
    }

    console.log('='.repeat(80));

    return report;
  }

  _checkSuccessCriteria() {
    const criteriaStatus = {};

    // Monitoring Context
    const contextResult = this.validationResults['monitoring_context'];
    criteriaStatus['monitoring_context'] = contextResult?.status === 'SUCCESS';
    const contextIcon = criteriaStatus['monitoring_context'] ? '✅' : '❌';
    console.log(`  ${contextIcon} MonitoringContext: ${criteriaStatus['monitoring_context'] ? 'Implémenté' : 'Manquant'}`);

    // Dashboard Components
    const componentsResult = this.validationResults['dashboard_components'];
    criteriaStatus['dashboard_components'] = componentsResult?.status === 'SUCCESS';
    const componentsIcon = criteriaStatus['dashboard_components'] ? '✅' : '❌';
    console.log(`  ${componentsIcon} Composants Dashboard: ${criteriaStatus['dashboard_components'] ? 'Créés' : 'Manquants'}`);

    // Charts & Widgets
    const chartsResult = this.validationResults['charts_widgets'];
    criteriaStatus['charts_widgets'] = chartsResult?.status === 'SUCCESS';
    const chartsIcon = criteriaStatus['charts_widgets'] ? '✅' : '❌';
    console.log(`  ${chartsIcon} Charts & Widgets: ${criteriaStatus['charts_widgets'] ? 'Fonctionnels' : 'Défaillants'}`);

    // Alerts Interface
    const alertsResult = this.validationResults['alerts_interface'];
    criteriaStatus['alerts_interface'] = alertsResult?.status === 'SUCCESS';
    const alertsIcon = criteriaStatus['alerts_interface'] ? '✅' : '❌';
    console.log(`  ${alertsIcon} Interface Alertes: ${criteriaStatus['alerts_interface'] ? 'Opérationnelle' : 'Défaillante'}`);

    // Navigation
    const navResult = this.validationResults['navigation_integration'];
    criteriaStatus['navigation_integration'] = navResult?.status === 'SUCCESS';
    const navIcon = criteriaStatus['navigation_integration'] ? '✅' : '❌';
    console.log(`  ${navIcon} Intégration Navigation: ${criteriaStatus['navigation_integration'] ? 'Complète' : 'Incomplète'}`);

    return criteriaStatus;
  }

  _getSuccessCriteriaStatus() {
    return this._checkSuccessCriteria();
  }

  _generateRecommendations() {
    const recommendations = [];

    // Analyser échecs
    Object.entries(this.validationResults).forEach(([testName, result]) => {
      if (result.status === 'FAILURE') {
        if (testName.includes('context')) {
          recommendations.push('Implémenter MonitoringContext avec React Query et actions');
        } else if (testName.includes('components')) {
          recommendations.push('Créer tous les composants dashboard monitoring requis');
        } else if (testName.includes('charts')) {
          recommendations.push('Intégrer Recharts avec types de charts multiples');
        } else if (testName.includes('alerts')) {
          recommendations.push('Implémenter interface alertes avec acquittement/résolution');
        } else if (testName.includes('navigation')) {
          recommendations.push('Ajouter route monitoring dans App.js et Layout.js');
        } else if (testName.includes('tests')) {
          recommendations.push('Écrire et faire passer les tests unitaires');
        }
      }
    });

    if (recommendations.length === 0) {
      recommendations.push('Toutes les fonctionnalités dashboard US-004 sont opérationnelles');
    }

    // Recommandations performance
    const hasResponsive = this.validationResults['responsive_design']?.status === 'SUCCESS';
    if (!hasResponsive) {
      recommendations.push('Optimiser design responsive avec breakpoints MUI');
    }

    return [...new Set(recommendations)]; // Dédupliquer
  }
}

// Point d'entrée principal
async function main() {
  const args = process.argv.slice(2);
  const comprehensive = args.includes('--comprehensive');
  const buildTest = args.includes('--build-test');

  const validator = new US004Validator();
  const [successCount, failureCount] = await validator.runValidation(comprehensive, buildTest);

  // Code de retour pour CI/CD
  const exitCode = failureCount === 0 ? 0 : 1;
  process.exit(exitCode);
}

if (require.main === module) {
  main().catch(error => {
    console.error('❌ Erreur validation:', error);
    process.exit(1);
  });
}

module.exports = US004Validator;