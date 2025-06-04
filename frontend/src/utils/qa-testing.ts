/**
 * Utilitaires de Tests QA - M&A Intelligence Platform
 * Sprint 6 - Tests automatisés pour responsive, accessibilité et compatibilité
 */

// Types pour les tests QA
export interface ResponsiveBreakpoint {
  name: string;
  width: number;
  height: number;
  description: string;
}

export interface AccessibilityIssue {
  level: 'error' | 'warning' | 'info';
  rule: string;
  element: string;
  description: string;
  suggestion: string;
}

export interface BrowserCompatibilityResult {
  browser: string;
  version: string;
  supported: boolean;
  issues: string[];
  features: Record<string, boolean>;
}

export interface QATestResult {
  category: 'responsive' | 'accessibility' | 'performance' | 'compatibility';
  test: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
  details?: any;
  timestamp: string;
}

// Breakpoints responsive standards
export const RESPONSIVE_BREAKPOINTS: ResponsiveBreakpoint[] = [
  { name: 'Mobile Portrait', width: 320, height: 568, description: 'iPhone SE' },
  { name: 'Mobile Landscape', width: 568, height: 320, description: 'iPhone SE Landscape' },
  { name: 'Tablet Portrait', width: 768, height: 1024, description: 'iPad Portrait' },
  { name: 'Tablet Landscape', width: 1024, height: 768, description: 'iPad Landscape' },
  { name: 'Desktop Small', width: 1280, height: 720, description: 'Small Desktop' },
  { name: 'Desktop Large', width: 1920, height: 1080, description: 'Large Desktop' },
  { name: 'Ultra Wide', width: 2560, height: 1440, description: 'Ultra Wide Monitor' }
];

// Tests de responsive design
export class ResponsiveDesignTester {
  private results: QATestResult[] = [];

  async testAllBreakpoints(): Promise<QATestResult[]> {
    this.results = [];

    for (const breakpoint of RESPONSIVE_BREAKPOINTS) {
      await this.testBreakpoint(breakpoint);
    }

    return this.results;
  }

  private async testBreakpoint(breakpoint: ResponsiveBreakpoint): Promise<void> {
    // Simuler le changement de viewport
    this.setViewport(breakpoint.width, breakpoint.height);
    
    // Attendre que le DOM se stabilise
    await this.waitForLayoutStabilization();

    // Tests spécifiques au breakpoint
    this.testNavigation(breakpoint);
    this.testContentVisibility(breakpoint);
    this.testInteractionElements(breakpoint);
    this.testScrollBehavior(breakpoint);
    this.testTextReadability(breakpoint);
  }

  private setViewport(width: number, height: number): void {
    // En environnement de test, utiliser une approche de simulation
    if (typeof window !== 'undefined') {
      Object.defineProperty(window, 'innerWidth', { value: width, writable: true });
      Object.defineProperty(window, 'innerHeight', { value: height, writable: true });
      window.dispatchEvent(new Event('resize'));
    }
  }

  private async waitForLayoutStabilization(): Promise<void> {
    return new Promise(resolve => {
      const observer = new ResizeObserver(() => {
        clearTimeout(timeout);
        timeout = setTimeout(() => {
          observer.disconnect();
          resolve();
        }, 100);
      });

      let timeout = setTimeout(() => {
        observer.disconnect();
        resolve();
      }, 500);

      observer.observe(document.body);
    });
  }

  private testNavigation(breakpoint: ResponsiveBreakpoint): void {
    const nav = document.querySelector('nav, [role="navigation"]');
    
    if (!nav) {
      this.addResult('responsive', 'Navigation Test', 'warning', 
        'No navigation element found', { breakpoint: breakpoint.name });
      return;
    }

    const isVisible = this.isElementVisible(nav);
    const isMobile = breakpoint.width < 768;

    if (isMobile) {
      // Sur mobile, vérifier la présence d'un menu burger
      const mobileToggle = nav.querySelector('[data-testid="mobile-menu-toggle"], .mobile-menu-toggle, button[aria-expanded]');
      
      if (!mobileToggle) {
        this.addResult('responsive', 'Mobile Navigation', 'fail',
          'Mobile menu toggle not found', { breakpoint: breakpoint.name });
      } else {
        this.addResult('responsive', 'Mobile Navigation', 'pass',
          'Mobile menu toggle found', { breakpoint: breakpoint.name });
      }
    } else {
      // Sur desktop, vérifier que la navigation est visible
      if (isVisible) {
        this.addResult('responsive', 'Desktop Navigation', 'pass',
          'Navigation is visible on desktop', { breakpoint: breakpoint.name });
      } else {
        this.addResult('responsive', 'Desktop Navigation', 'fail',
          'Navigation is not visible on desktop', { breakpoint: breakpoint.name });
      }
    }
  }

  private testContentVisibility(breakpoint: ResponsiveBreakpoint): void {
    const mainContent = document.querySelector('main, [role="main"], .main-content');
    
    if (!mainContent) {
      this.addResult('responsive', 'Main Content', 'warning',
        'Main content area not found', { breakpoint: breakpoint.name });
      return;
    }

    const isVisible = this.isElementVisible(mainContent);
    const hasOverflow = this.hasHorizontalOverflow(mainContent);

    if (!isVisible) {
      this.addResult('responsive', 'Content Visibility', 'fail',
        'Main content is not visible', { breakpoint: breakpoint.name });
    } else if (hasOverflow) {
      this.addResult('responsive', 'Content Overflow', 'warning',
        'Content has horizontal overflow', { breakpoint: breakpoint.name });
    } else {
      this.addResult('responsive', 'Content Layout', 'pass',
        'Content is properly visible and contained', { breakpoint: breakpoint.name });
    }
  }

  private testInteractionElements(breakpoint: ResponsiveBreakpoint): void {
    const buttons = document.querySelectorAll('button, [role="button"], a[href]');
    const isMobile = breakpoint.width < 768;
    const minTouchTarget = 44; // 44px minimum pour les éléments tactiles

    let failedElements = 0;

    buttons.forEach((element, index) => {
      const rect = element.getBoundingClientRect();
      const size = Math.min(rect.width, rect.height);

      if (isMobile && size < minTouchTarget) {
        failedElements++;
      }
    });

    if (isMobile && failedElements > 0) {
      this.addResult('responsive', 'Touch Targets', 'warning',
        `${failedElements} elements are too small for touch interaction`,
        { breakpoint: breakpoint.name, failedElements });
    } else {
      this.addResult('responsive', 'Touch Targets', 'pass',
        'All interactive elements meet size requirements',
        { breakpoint: breakpoint.name });
    }
  }

  private testScrollBehavior(breakpoint: ResponsiveBreakpoint): void {
    const hasHorizontalScroll = document.body.scrollWidth > window.innerWidth;
    
    if (hasHorizontalScroll) {
      this.addResult('responsive', 'Scroll Behavior', 'warning',
        'Horizontal scroll detected', { breakpoint: breakpoint.name });
    } else {
      this.addResult('responsive', 'Scroll Behavior', 'pass',
        'No unwanted horizontal scroll', { breakpoint: breakpoint.name });
    }
  }

  private testTextReadability(breakpoint: ResponsiveBreakpoint): void {
    const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, li');
    let readabilityIssues = 0;

    textElements.forEach(element => {
      const styles = window.getComputedStyle(element);
      const fontSize = parseFloat(styles.fontSize);
      const lineHeight = parseFloat(styles.lineHeight) || fontSize * 1.2;
      
      // Vérifier la taille minimale de police
      if (fontSize < 14) {
        readabilityIssues++;
      }
      
      // Vérifier l'espacement des lignes
      if (lineHeight / fontSize < 1.2) {
        readabilityIssues++;
      }
    });

    if (readabilityIssues > 0) {
      this.addResult('responsive', 'Text Readability', 'warning',
        `${readabilityIssues} text elements have readability issues`,
        { breakpoint: breakpoint.name, issues: readabilityIssues });
    } else {
      this.addResult('responsive', 'Text Readability', 'pass',
        'Text readability is good', { breakpoint: breakpoint.name });
    }
  }

  private isElementVisible(element: Element): boolean {
    const rect = element.getBoundingClientRect();
    const styles = window.getComputedStyle(element);
    
    return rect.width > 0 && 
           rect.height > 0 && 
           styles.visibility !== 'hidden' && 
           styles.display !== 'none';
  }

  private hasHorizontalOverflow(element: Element): boolean {
    return element.scrollWidth > element.clientWidth;
  }

  private addResult(category: QATestResult['category'], test: string, status: QATestResult['status'], message: string, details?: any): void {
    this.results.push({
      category,
      test,
      status,
      message,
      details,
      timestamp: new Date().toISOString()
    });
  }
}

// Tests d'accessibilité
export class AccessibilityTester {
  private results: QATestResult[] = [];

  async runAllTests(): Promise<QATestResult[]> {
    this.results = [];

    this.testKeyboardNavigation();
    this.testAriaLabels();
    this.testColorContrast();
    this.testSemanticStructure();
    this.testFocusManagement();
    this.testScreenReaderCompatibility();
    this.testAlternativeText();

    return this.results;
  }

  private testKeyboardNavigation(): void {
    const interactiveElements = document.querySelectorAll(
      'button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    let nonFocusableElements = 0;

    interactiveElements.forEach(element => {
      const tabIndex = element.getAttribute('tabindex');
      const isNaturallyFocusable = ['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName);
      
      if (!isNaturallyFocusable && tabIndex === null) {
        nonFocusableElements++;
      }
    });

    if (nonFocusableElements > 0) {
      this.addResult('accessibility', 'Keyboard Navigation', 'warning',
        `${nonFocusableElements} interactive elements may not be keyboard accessible`,
        { nonFocusableElements });
    } else {
      this.addResult('accessibility', 'Keyboard Navigation', 'pass',
        'All interactive elements are keyboard accessible');
    }
  }

  private testAriaLabels(): void {
    const elementsNeedingLabels = document.querySelectorAll(
      'button:not([aria-label]):not([aria-labelledby]), input:not([aria-label]):not([aria-labelledby]):not([id])'
    );

    if (elementsNeedingLabels.length > 0) {
      this.addResult('accessibility', 'ARIA Labels', 'warning',
        `${elementsNeedingLabels.length} elements missing accessibility labels`,
        { missingLabels: elementsNeedingLabels.length });
    } else {
      this.addResult('accessibility', 'ARIA Labels', 'pass',
        'All elements have proper accessibility labels');
    }
  }

  private testColorContrast(): void {
    // Test simplifié - en production, utiliser une librairie comme axe-core
    const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, a, button');
    let lowContrastElements = 0;

    textElements.forEach(element => {
      const styles = window.getComputedStyle(element);
      const color = styles.color;
      const backgroundColor = styles.backgroundColor;
      
      // Simulation simple - en réalité, il faut calculer le ratio de contraste
      if (this.isLowContrast(color, backgroundColor)) {
        lowContrastElements++;
      }
    });

    if (lowContrastElements > 0) {
      this.addResult('accessibility', 'Color Contrast', 'warning',
        `${lowContrastElements} elements may have insufficient color contrast`,
        { lowContrastElements });
    } else {
      this.addResult('accessibility', 'Color Contrast', 'pass',
        'Color contrast appears adequate');
    }
  }

  private testSemanticStructure(): void {
    const hasH1 = document.querySelector('h1') !== null;
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const hasMain = document.querySelector('main, [role="main"]') !== null;
    const hasNav = document.querySelector('nav, [role="navigation"]') !== null;

    if (!hasH1) {
      this.addResult('accessibility', 'Semantic Structure', 'warning',
        'No H1 heading found on page');
    } else if (!hasMain) {
      this.addResult('accessibility', 'Semantic Structure', 'warning',
        'No main content area identified');
    } else if (!hasNav) {
      this.addResult('accessibility', 'Semantic Structure', 'warning',
        'No navigation area identified');
    } else {
      this.addResult('accessibility', 'Semantic Structure', 'pass',
        'Good semantic structure with proper landmarks');
    }
  }

  private testFocusManagement(): void {
    const activeElement = document.activeElement;
    const focusableElements = document.querySelectorAll(
      'button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    // Vérifier si les éléments ont des styles de focus visibles
    let elementsWithoutFocusStyle = 0;

    focusableElements.forEach(element => {
      const styles = window.getComputedStyle(element, ':focus');
      const hasOutline = styles.outline !== 'none' && styles.outline !== '0px';
      const hasBoxShadow = styles.boxShadow !== 'none';
      
      if (!hasOutline && !hasBoxShadow) {
        elementsWithoutFocusStyle++;
      }
    });

    if (elementsWithoutFocusStyle > 0) {
      this.addResult('accessibility', 'Focus Management', 'warning',
        `${elementsWithoutFocusStyle} elements lack visible focus indicators`,
        { elementsWithoutFocusStyle });
    } else {
      this.addResult('accessibility', 'Focus Management', 'pass',
        'All focusable elements have visible focus indicators');
    }
  }

  private testScreenReaderCompatibility(): void {
    const ariaLiveRegions = document.querySelectorAll('[aria-live]');
    const skipLinks = document.querySelectorAll('a[href^="#"]:first-child');
    const landmarks = document.querySelectorAll('main, nav, aside, header, footer, [role]');

    let score = 0;
    const maxScore = 3;

    if (ariaLiveRegions.length > 0) score++;
    if (skipLinks.length > 0) score++;
    if (landmarks.length >= 3) score++;

    if (score === maxScore) {
      this.addResult('accessibility', 'Screen Reader Support', 'pass',
        'Good screen reader compatibility features detected');
    } else {
      this.addResult('accessibility', 'Screen Reader Support', 'warning',
        `Screen reader compatibility could be improved (${score}/${maxScore})`,
        { score, maxScore });
    }
  }

  private testAlternativeText(): void {
    const images = document.querySelectorAll('img');
    let imagesWithoutAlt = 0;

    images.forEach(img => {
      const alt = img.getAttribute('alt');
      if (alt === null || alt === '') {
        imagesWithoutAlt++;
      }
    });

    if (imagesWithoutAlt > 0) {
      this.addResult('accessibility', 'Alternative Text', 'warning',
        `${imagesWithoutAlt} images missing alternative text`,
        { imagesWithoutAlt });
    } else {
      this.addResult('accessibility', 'Alternative Text', 'pass',
        'All images have alternative text');
    }
  }

  private isLowContrast(color: string, backgroundColor: string): boolean {
    // Implémentation simplifiée - en production, utiliser une vraie fonction de calcul de contraste
    return color === backgroundColor || 
           (color.includes('rgb(128') && backgroundColor.includes('rgb(128'));
  }

  private addResult(category: QATestResult['category'], test: string, status: QATestResult['status'], message: string, details?: any): void {
    this.results.push({
      category,
      test,
      status,
      message,
      details,
      timestamp: new Date().toISOString()
    });
  }
}

// Tests de compatibilité navigateur
export class BrowserCompatibilityTester {
  private results: QATestResult[] = [];

  async runCompatibilityTests(): Promise<QATestResult[]> {
    this.results = [];

    this.testCSSFeatures();
    this.testJavaScriptFeatures();
    this.testAPISupport();
    this.testPerformanceAPIs();

    return this.results;
  }

  private testCSSFeatures(): void {
    const features = {
      'CSS Grid': this.supportsCSSGrid(),
      'CSS Flexbox': this.supportsCSSFlexbox(),
      'CSS Custom Properties': this.supportsCSSCustomProperties(),
      'CSS Animations': this.supportsCSSAnimations(),
      'CSS Transforms': this.supportsCSSTransforms()
    };

    let unsupportedFeatures = 0;
    const featureList: string[] = [];

    Object.entries(features).forEach(([feature, supported]) => {
      if (!supported) {
        unsupportedFeatures++;
        featureList.push(feature);
      }
    });

    if (unsupportedFeatures > 0) {
      this.addResult('compatibility', 'CSS Features', 'warning',
        `${unsupportedFeatures} CSS features not supported`,
        { unsupportedFeatures: featureList });
    } else {
      this.addResult('compatibility', 'CSS Features', 'pass',
        'All required CSS features are supported');
    }
  }

  private testJavaScriptFeatures(): void {
    const features = {
      'ES6 Modules': typeof Symbol !== 'undefined',
      'Async/Await': this.supportsAsyncAwait(),
      'Promises': typeof Promise !== 'undefined',
      'Arrow Functions': this.supportsArrowFunctions(),
      'Template Literals': this.supportsTemplateLiterals(),
      'Destructuring': this.supportsDestructuring()
    };

    let unsupportedFeatures = 0;
    const featureList: string[] = [];

    Object.entries(features).forEach(([feature, supported]) => {
      if (!supported) {
        unsupportedFeatures++;
        featureList.push(feature);
      }
    });

    if (unsupportedFeatures > 0) {
      this.addResult('compatibility', 'JavaScript Features', 'fail',
        `${unsupportedFeatures} JavaScript features not supported`,
        { unsupportedFeatures: featureList });
    } else {
      this.addResult('compatibility', 'JavaScript Features', 'pass',
        'All required JavaScript features are supported');
    }
  }

  private testAPISupport(): void {
    const apis = {
      'Fetch API': typeof fetch !== 'undefined',
      'Intersection Observer': typeof IntersectionObserver !== 'undefined',
      'ResizeObserver': typeof ResizeObserver !== 'undefined',
      'MutationObserver': typeof MutationObserver !== 'undefined',
      'Web Workers': typeof Worker !== 'undefined',
      'Service Workers': 'serviceWorker' in navigator,
      'Local Storage': typeof localStorage !== 'undefined',
      'Session Storage': typeof sessionStorage !== 'undefined'
    };

    let unsupportedAPIs = 0;
    const apiList: string[] = [];

    Object.entries(apis).forEach(([api, supported]) => {
      if (!supported) {
        unsupportedAPIs++;
        apiList.push(api);
      }
    });

    if (unsupportedAPIs > 0) {
      this.addResult('compatibility', 'Web APIs', 'warning',
        `${unsupportedAPIs} Web APIs not supported`,
        { unsupportedAPIs: apiList });
    } else {
      this.addResult('compatibility', 'Web APIs', 'pass',
        'All required Web APIs are supported');
    }
  }

  private testPerformanceAPIs(): void {
    const performanceFeatures = {
      'Performance API': 'performance' in window,
      'Performance Observer': typeof PerformanceObserver !== 'undefined',
      'Navigation Timing': 'navigation' in performance,
      'Resource Timing': 'getEntriesByType' in performance,
      'User Timing': 'mark' in performance && 'measure' in performance
    };

    let unsupportedFeatures = 0;
    const featureList: string[] = [];

    Object.entries(performanceFeatures).forEach(([feature, supported]) => {
      if (!supported) {
        unsupportedFeatures++;
        featureList.push(feature);
      }
    });

    if (unsupportedFeatures > 0) {
      this.addResult('compatibility', 'Performance APIs', 'warning',
        `${unsupportedFeatures} Performance APIs not supported`,
        { unsupportedFeatures: featureList });
    } else {
      this.addResult('compatibility', 'Performance APIs', 'pass',
        'All Performance APIs are supported');
    }
  }

  // Méthodes de détection des fonctionnalités
  private supportsCSSGrid(): boolean {
    return CSS.supports('display', 'grid');
  }

  private supportsCSSFlexbox(): boolean {
    return CSS.supports('display', 'flex');
  }

  private supportsCSSCustomProperties(): boolean {
    return CSS.supports('--custom-property', 'value');
  }

  private supportsCSSAnimations(): boolean {
    return CSS.supports('animation-name', 'test');
  }

  private supportsCSSTransforms(): boolean {
    return CSS.supports('transform', 'translateX(1px)');
  }

  private supportsAsyncAwait(): boolean {
    try {
      // Test si async/await est supporté
      eval('(async function() {})');
      return true;
    } catch {
      return false;
    }
  }

  private supportsArrowFunctions(): boolean {
    try {
      eval('(() => {})');
      return true;
    } catch {
      return false;
    }
  }

  private supportsTemplateLiterals(): boolean {
    try {
      eval('`template literal`');
      return true;
    } catch {
      return false;
    }
  }

  private supportsDestructuring(): boolean {
    try {
      eval('const [a] = [1]');
      return true;
    } catch {
      return false;
    }
  }

  private addResult(category: QATestResult['category'], test: string, status: QATestResult['status'], message: string, details?: any): void {
    this.results.push({
      category,
      test,
      status,
      message,
      details,
      timestamp: new Date().toISOString()
    });
  }
}

// Classe principale pour orchestrer tous les tests QA
export class QATestSuite {
  private responsiveTester = new ResponsiveDesignTester();
  private accessibilityTester = new AccessibilityTester();
  private compatibilityTester = new BrowserCompatibilityTester();

  async runAllTests(): Promise<{
    responsive: QATestResult[];
    accessibility: QATestResult[];
    compatibility: QATestResult[];
    summary: {
      total: number;
      passed: number;
      failed: number;
      warnings: number;
    };
  }> {
    console.log('🔍 Starting comprehensive QA test suite...');

    const [responsive, accessibility, compatibility] = await Promise.all([
      this.responsiveTester.testAllBreakpoints(),
      this.accessibilityTester.runAllTests(),
      this.compatibilityTester.runCompatibilityTests()
    ]);

    const allResults = [...responsive, ...accessibility, ...compatibility];
    
    const summary = {
      total: allResults.length,
      passed: allResults.filter(r => r.status === 'pass').length,
      failed: allResults.filter(r => r.status === 'fail').length,
      warnings: allResults.filter(r => r.status === 'warning').length
    };

    console.log('✅ QA test suite completed:', summary);

    return {
      responsive,
      accessibility,
      compatibility,
      summary
    };
  }

  generateReport(results: any): string {
    const { responsive, accessibility, compatibility, summary } = results;
    
    let report = `# QA Test Report - ${new Date().toLocaleDateString()}\n\n`;
    report += `## Summary\n`;
    report += `- Total Tests: ${summary.total}\n`;
    report += `- Passed: ${summary.passed} ✅\n`;
    report += `- Failed: ${summary.failed} ❌\n`;
    report += `- Warnings: ${summary.warnings} ⚠️\n\n`;

    // Ajouter les détails par catégorie
    report += this.formatCategoryResults('Responsive Design', responsive);
    report += this.formatCategoryResults('Accessibility', accessibility);
    report += this.formatCategoryResults('Browser Compatibility', compatibility);

    return report;
  }

  private formatCategoryResults(category: string, results: QATestResult[]): string {
    let section = `## ${category}\n\n`;
    
    results.forEach(result => {
      const icon = result.status === 'pass' ? '✅' : result.status === 'fail' ? '❌' : '⚠️';
      section += `${icon} **${result.test}**: ${result.message}\n`;
      
      if (result.details) {
        section += `   - Details: ${JSON.stringify(result.details)}\n`;
      }
      section += '\n';
    });

    return section;
  }
}

export default {
  ResponsiveDesignTester,
  AccessibilityTester,
  BrowserCompatibilityTester,
  QATestSuite,
  RESPONSIVE_BREAKPOINTS
};