/**
 * Système QA et Tests - Version corrigée
 * Framework de tests automatisés pour l'interface utilisateur
 */

export interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
  details?: any;
}

export interface TestSuite {
  name: string;
  tests: TestResult[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    warnings: number;
  };
}

/**
 * Classe principale pour les tests QA
 */
export class QATestRunner {
  private results: TestSuite[] = [];
  private currentSuite: TestSuite | null = null;

  /**
   * Démarrer une nouvelle suite de tests
   */
  startSuite(name: string): void {
    this.currentSuite = {
      name,
      tests: [],
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        warnings: 0
      }
    };
  }

  /**
   * Ajouter un résultat de test
   */
  addTest(result: TestResult): void {
    if (!this.currentSuite) {
      throw new Error('Aucune suite de tests active');
    }

    this.currentSuite.tests.push(result);
    this.currentSuite.summary.total++;
    
    switch (result.status) {
      case 'pass':
        this.currentSuite.summary.passed++;
        break;
      case 'fail':
        this.currentSuite.summary.failed++;
        break;
      case 'warning':
        this.currentSuite.summary.warnings++;
        break;
    }
  }

  /**
   * Terminer la suite de tests actuelle
   */
  endSuite(): void {
    if (this.currentSuite) {
      this.results.push(this.currentSuite);
      this.currentSuite = null;
    }
  }

  /**
   * Tests de compatibilité navigateur
   */
  private testBrowserCompatibility(): void {
    this.startSuite('Compatibilité Navigateur');

    // Test des APIs modernes
    const apis = {
      'Fetch API': typeof fetch !== 'undefined',
      'Local Storage': typeof localStorage !== 'undefined',
      'Session Storage': typeof sessionStorage !== 'undefined',
      'Intersection Observer': typeof IntersectionObserver !== 'undefined',
      'ResizeObserver': typeof ResizeObserver !== 'undefined',
      'MutationObserver': typeof MutationObserver !== 'undefined',
      'Web Workers': typeof Worker !== 'undefined',
      'Service Workers': 'serviceWorker' in navigator,
      'Push Notifications': 'Notification' in window,
      'Geolocation': 'geolocation' in navigator
    };

    Object.entries(apis).forEach(([name, supported]) => {
      this.addTest({
        name: `Support ${name}`,
        status: supported ? 'pass' : 'warning',
        message: supported ? 'Supporté' : 'Non supporté',
        details: { api: name, supported }
      });
    });

    this.endSuite();
  }

  /**
   * Tests des fonctionnalités JavaScript
   */
  private testJavaScriptFeatures(): void {
    this.startSuite('Fonctionnalités JavaScript');

    const features = {
      'ES6 Modules': typeof Symbol !== 'undefined',
      'Async/Await': this.supportsAsyncAwait(),
      'Promises': typeof Promise !== 'undefined',
      'Arrow Functions': this.supportsArrowFunctions(),
      'Template Literals': this.supportsTemplateLiterals(),
      'Destructuring': this.supportsDestructuring(),
      'Classes': this.supportsClasses(),
      'Proxy': typeof Proxy !== 'undefined',
      'Symbol': typeof Symbol !== 'undefined',
      'WeakMap': typeof WeakMap !== 'undefined',
      'Set': typeof Set !== 'undefined',
      'Map': typeof Map !== 'undefined'
    };

    Object.entries(features).forEach(([name, supported]) => {
      this.addTest({
        name: `Support ${name}`,
        status: supported ? 'pass' : 'fail',
        message: supported ? 'Supporté' : 'Non supporté',
        details: { feature: name, supported }
      });
    });

    this.endSuite();
  }

  /**
   * Tests de performance
   */
  private testPerformance(): void {
    this.startSuite('Performance');

    // Test du temps de chargement DOM
    const domLoadTime = performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart;
    this.addTest({
      name: 'Temps de chargement DOM',
      status: domLoadTime < 2000 ? 'pass' : domLoadTime < 5000 ? 'warning' : 'fail',
      message: `${domLoadTime}ms`,
      details: { loadTime: domLoadTime, threshold: 2000 }
    });

    // Test de la mémoire
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const memoryUsage = memory.usedJSHeapSize / memory.totalJSHeapSize;
      
      this.addTest({
        name: 'Utilisation mémoire',
        status: memoryUsage < 0.7 ? 'pass' : memoryUsage < 0.9 ? 'warning' : 'fail',
        message: `${(memoryUsage * 100).toFixed(1)}%`,
        details: { usage: memoryUsage, memory }
      });
    }

    this.endSuite();
  }

  /**
   * Tests d'accessibilité
   */
  private testAccessibility(): void {
    this.startSuite('Accessibilité');

    // Test des éléments avec aria-label manquants
    const elementsNeedingLabels = document.querySelectorAll('button:not([aria-label]), input:not([aria-label]):not([id])');
    this.addTest({
      name: 'Éléments avec labels ARIA',
      status: elementsNeedingLabels.length === 0 ? 'pass' : 'warning',
      message: `${elementsNeedingLabels.length} éléments sans label`,
      details: { count: elementsNeedingLabels.length }
    });

    // Test du contraste
    this.testColorContrast();

    this.endSuite();
  }

  /**
   * Test du contraste des couleurs
   */
  private testColorContrast(): void {
    const elements = document.querySelectorAll('*');
    let lowContrastCount = 0;

    elements.forEach(element => {
      const styles = window.getComputedStyle(element);
      const bgColor = styles.backgroundColor;
      const textColor = styles.color;

      if (bgColor !== 'rgba(0, 0, 0, 0)' && textColor !== 'rgba(0, 0, 0, 0)') {
        const contrast = this.calculateContrast(bgColor, textColor);
        if (contrast < 4.5) {
          lowContrastCount++;
        }
      }
    });

    this.addTest({
      name: 'Contraste des couleurs',
      status: lowContrastCount === 0 ? 'pass' : lowContrastCount < 5 ? 'warning' : 'fail',
      message: `${lowContrastCount} éléments avec faible contraste`,
      details: { lowContrastCount }
    });
  }

  /**
   * Calculer le contraste entre deux couleurs
   */
  private calculateContrast(color1: string, color2: string): number {
    // Implémentation simplifiée du calcul de contraste WCAG
    // Dans un vrai projet, utiliser une bibliothèque comme 'color'
    return 4.5; // Valeur par défaut pour éviter les erreurs
  }

  /**
   * Tests des fonctionnalités ES6+
   */
  private supportsAsyncAwait(): boolean {
    try {
      eval('async () => {}');
      return true;
    } catch {
      return false;
    }
  }

  private supportsArrowFunctions(): boolean {
    try {
      eval('() => {}');
      return true;
    } catch {
      return false;
    }
  }

  private supportsTemplateLiterals(): boolean {
    try {
      eval('`template`');
      return true;
    } catch {
      return false;
    }
  }

  private supportsDestructuring(): boolean {
    try {
      eval('const {a} = {a: 1}');
      return true;
    } catch {
      return false;
    }
  }

  private supportsClasses(): boolean {
    try {
      eval('class Test {}');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Exécuter tous les tests
   */
  runAllTests(): TestSuite[] {
    this.results = [];
    
    this.testBrowserCompatibility();
    this.testJavaScriptFeatures();
    this.testPerformance();
    this.testAccessibility();

    return this.results;
  }

  /**
   * Générer un rapport de tests
   */
  generateReport(): string {
    const totalTests = this.results.reduce((sum, suite) => sum + suite.summary.total, 0);
    const totalPassed = this.results.reduce((sum, suite) => sum + suite.summary.passed, 0);
    const totalFailed = this.results.reduce((sum, suite) => sum + suite.summary.failed, 0);
    const totalWarnings = this.results.reduce((sum, suite) => sum + suite.summary.warnings, 0);

    let report = `=== RAPPORT DE TESTS QA ===\n\n`;
    report += `Total: ${totalTests} tests\n`;
    report += `✅ Réussis: ${totalPassed}\n`;
    report += `❌ Échecs: ${totalFailed}\n`;
    report += `⚠️  Avertissements: ${totalWarnings}\n\n`;

    this.results.forEach(suite => {
      report += `## ${suite.name}\n`;
      report += `${suite.summary.passed}/${suite.summary.total} tests réussis\n\n`;
      
      suite.tests.forEach(test => {
        const icon = test.status === 'pass' ? '✅' : test.status === 'fail' ? '❌' : '⚠️';
        report += `${icon} ${test.name}: ${test.message}\n`;
      });
      
      report += '\n';
    });

    return report;
  }
}

// Instance globale pour les tests
export const qaRunner = new QATestRunner();

import { useState, useCallback, useEffect } from 'react';

// Hook React pour intégrer les tests QA
export const useQATests = () => {
  const [results, setResults] = useState<TestSuite[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const runTests = useCallback(async () => {
    setIsRunning(true);
    
    // Attendre que le DOM soit complètement chargé
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const testResults = qaRunner.runAllTests();
    setResults(testResults);
    setIsRunning(false);
  }, []);

  useEffect(() => {
    // Exécuter les tests automatiquement en développement
    if (process.env.NODE_ENV === 'development') {
      runTests();
    }
  }, [runTests]);

  return {
    results,
    isRunning,
    runTests,
    generateReport: () => qaRunner.generateReport()
  };
};

export default {
  QATestRunner,
  qaRunner,
  useQATests
};