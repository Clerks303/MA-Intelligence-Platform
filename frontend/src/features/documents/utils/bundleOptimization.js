/**
 * Bundle Optimization - M&A Intelligence Platform
 * Sprint 3 - Configuration optimisation bundle et code splitting
 */

// Configuration pour optimiser le bundling du module documents
export const documentsBundleConfig = {
  // Code splitting par feature
  chunks: {
    // Chunk principal du module documents
    documents: {
      name: 'documents',
      chunks: 'async',
      test: /src[\\/]features[\\/]documents/,
      minChunks: 1,
      maxAsyncRequests: 5,
      maxInitialRequests: 3,
      enforce: true,
    },
    
    // Chunk pour les composants de preview
    'documents-preview': {
      name: 'documents-preview',
      chunks: 'async',
      test: /src[\\/]features[\\/]documents[\\/]components[\\/]previews/,
      minChunks: 1,
      priority: 10,
    },
    
    // Chunk pour les services
    'documents-services': {
      name: 'documents-services',
      chunks: 'async',
      test: /src[\\/]features[\\/]documents[\\/]services/,
      minChunks: 1,
      priority: 5,
    },
    
    // Chunk pour les utilitaires
    'documents-utils': {
      name: 'documents-utils',
      chunks: 'async',
      test: /src[\\/]features[\\/]documents[\\/]utils/,
      minChunks: 1,
      priority: 1,
    },
  },

  // Tree shaking configuration
  treeShaking: {
    usedExports: true,
    sideEffects: false,
    optimization: {
      // Marquer les exports non utilisés
      providedExports: true,
      usedExports: true,
      
      // Concaténation des modules
      concatenateModules: true,
      
      // Minification
      minimize: true,
      minimizer: [
        // Configuration TerserPlugin pour documents
        {
          test: /\.js(\?.*)?$/i,
          include: /src[\\/]features[\\/]documents/,
          terserOptions: {
            compress: {
              drop_console: true,
              drop_debugger: true,
              pure_funcs: ['console.log', 'console.info'],
            },
            mangle: {
              safari10: true,
            },
            output: {
              comments: false,
              ascii_only: true,
            },
          },
        },
      ],
    },
  },

  // Analyse bundle
  analysis: {
    // Tailles cibles (en KB)
    targets: {
      'documents.js': 200,
      'documents-preview.js': 150,
      'documents-services.js': 100,
      'documents-utils.js': 50,
    },
    
    // Seuils d'alerte
    thresholds: {
      warning: 300, // KB
      error: 500,   // KB
    },
  },

  // Préchargement intelligent
  preload: {
    // Précharger les chunks critiques
    critical: ['documents'],
    
    // Préchargement conditionnel
    conditional: {
      'documents-preview': 'hover', // Précharger au survol
      'documents-services': 'idle',  // Précharger quand idle
    },
  },
};

// Fonction pour générer la configuration webpack
export const generateWebpackConfig = (env = 'production') => {
  const isDev = env === 'development';
  
  return {
    optimization: {
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          ...documentsBundleConfig.chunks,
          
          // Vendor chunk séparé
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
            priority: 20,
          },
        },
      },
      
      // Code splitting automatique
      runtimeChunk: 'single',
      
      // Minification conditionnelle
      minimize: !isDev,
      minimizer: isDev ? [] : documentsBundleConfig.treeShaking.optimization.minimizer,
    },
    
    // Configuration pour le développement
    ...(isDev && {
      devtool: 'eval-source-map',
      cache: {
        type: 'filesystem',
        buildDependencies: {
          config: [__filename],
        },
      },
    }),
    
    // Configuration pour la production
    ...(!isDev && {
      devtool: 'source-map',
      performance: {
        hints: 'warning',
        maxEntrypointSize: 512000, // 500KB
        maxAssetSize: 256000,      // 250KB
      },
    }),
  };
};

// Utilitaire d'analyse des bundles
export const bundleAnalyzer = {
  // Analyser la taille des chunks
  analyzeSizes: (stats) => {
    const assets = stats.compilation.assets;
    const analysis = {};
    
    Object.keys(assets).forEach(assetName => {
      if (assetName.includes('documents')) {
        const size = assets[assetName].size();
        const sizeKB = Math.round(size / 1024);
        
        analysis[assetName] = {
          size: sizeKB,
          target: documentsBundleConfig.analysis.targets[assetName] || 'N/A',
          status: sizeKB > documentsBundleConfig.analysis.thresholds.error ? 'error' :
                  sizeKB > documentsBundleConfig.analysis.thresholds.warning ? 'warning' : 'ok'
        };
      }
    });
    
    return analysis;
  },
  
  // Rapport de performance
  generateReport: (analysis) => {
    console.log('\n📊 Documents Module Bundle Analysis:');
    console.log('=====================================');
    
    Object.entries(analysis).forEach(([chunk, data]) => {
      const status = data.status === 'ok' ? '✅' : 
                    data.status === 'warning' ? '⚠️' : '❌';
      
      console.log(`${status} ${chunk}: ${data.size}KB (target: ${data.target}KB)`);
    });
    
    const totalSize = Object.values(analysis).reduce((sum, data) => sum + data.size, 0);
    console.log(`\n📦 Total Documents Module Size: ${totalSize}KB`);
    
    // Recommandations
    const issues = Object.entries(analysis).filter(([, data]) => data.status !== 'ok');
    if (issues.length > 0) {
      console.log('\n🔧 Recommendations:');
      issues.forEach(([chunk, data]) => {
        if (data.status === 'error') {
          console.log(`  • ${chunk}: Consider code splitting or lazy loading`);
        } else {
          console.log(`  • ${chunk}: Monitor size growth`);
        }
      });
    }
  },
};

// Plugin webpack personnalisé pour l'optimisation documents
export class DocumentsOptimizationPlugin {
  apply(compiler) {
    compiler.hooks.emit.tapAsync('DocumentsOptimizationPlugin', (compilation, callback) => {
      // Analyser les bundles
      const analysis = bundleAnalyzer.analyzeSizes({ compilation });
      
      // Générer le rapport en mode développement
      if (compilation.options.mode === 'development') {
        bundleAnalyzer.generateReport(analysis);
      }
      
      // Ajouter les métadonnées d'optimisation
      const optimizationData = JSON.stringify({
        timestamp: new Date().toISOString(),
        analysis,
        config: documentsBundleConfig,
      }, null, 2);
      
      compilation.assets['documents-optimization.json'] = {
        source: () => optimizationData,
        size: () => optimizationData.length,
      };
      
      callback();
    });
  }
}

// Configuration ESLint pour l'optimisation
export const eslintOptimizationConfig = {
  rules: {
    // Forcer l'utilisation de lazy loading pour les gros composants
    'documents/prefer-lazy-loading': {
      selector: 'ImportDeclaration[source.value=/components\\/.*Preview/]',
      message: 'Use lazy loading for preview components to optimize bundle size',
    },
    
    // Éviter les imports barrel non optimisés
    'documents/no-barrel-imports': {
      selector: 'ImportDeclaration[source.value=/features\\/documents$/]',
      message: 'Avoid barrel imports, use specific imports to enable tree shaking',
    },
    
    // Forcer l'utilisation du cache de performance
    'documents/prefer-performance-cache': {
      selector: 'CallExpression[callee.name="fetch"]',
      message: 'Use performanceCache for API calls to improve performance',
    },
  },
};

export default {
  bundleConfig: documentsBundleConfig,
  webpackConfig: generateWebpackConfig,
  analyzer: bundleAnalyzer,
  plugin: DocumentsOptimizationPlugin,
  eslintConfig: eslintOptimizationConfig,
};