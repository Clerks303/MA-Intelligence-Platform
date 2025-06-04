/**
 * Optimisation du Build Final - M&A Intelligence Platform
 * Sprint 6 - Configuration et optimisations pour build de production
 */

// Types pour l'optimisation du build
export interface BuildConfig {
  environment: 'development' | 'production' | 'staging';
  outputPath: string;
  publicPath: string;
  sourceMaps: boolean;
  minification: boolean;
  compression: boolean;
  bundleAnalysis: boolean;
  progressiveWebApp: boolean;
}

export interface BundleAnalysisResult {
  totalSize: number;
  gzippedSize: number;
  chunks: Array<{
    name: string;
    size: number;
    modules: number;
  }>;
  assets: Array<{
    name: string;
    size: number;
    type: string;
  }>;
  dependencies: Record<string, number>;
}

export interface PerformanceBudget {
  maxInitialBundleSize: number;
  maxChunkSize: number;
  maxAssetSize: number;
  maxEntrySize: number;
}

// Configuration par défaut pour la production
export const PRODUCTION_BUILD_CONFIG: BuildConfig = {
  environment: 'production',
  outputPath: 'dist',
  publicPath: '/',
  sourceMaps: false,
  minification: true,
  compression: true,
  bundleAnalysis: true,
  progressiveWebApp: true
};

// Budget de performance pour les bundles
export const PERFORMANCE_BUDGET: PerformanceBudget = {
  maxInitialBundleSize: 500 * 1024, // 500KB
  maxChunkSize: 200 * 1024,         // 200KB
  maxAssetSize: 100 * 1024,         // 100KB
  maxEntrySize: 300 * 1024          // 300KB
};

// Configuration webpack optimisée pour la production
export const getOptimizedWebpackConfig = (config: BuildConfig) => {
  return {
    mode: config.environment,
    
    // Optimisations
    optimization: {
      minimize: config.minification,
      minimizer: [
        // TerserPlugin pour JS
        {
          terserOptions: {
            compress: {
              drop_console: config.environment === 'production',
              drop_debugger: true,
              pure_funcs: ['console.log', 'console.info', 'console.debug']
            },
            mangle: {
              safari10: true
            },
            output: {
              comments: false,
              ascii_only: true
            }
          }
        },
        
        // CSS optimization
        {
          cssProcessorOptions: {
            map: config.sourceMaps ? {
              inline: false,
              annotation: true
            } : false
          }
        }
      ],
      
      // Split chunks de manière optimale
      splitChunks: {
        chunks: 'all',
        maxInitialRequests: 25,
        maxAsyncRequests: 25,
        cacheGroups: {
          // Vendor chunks
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 10,
            reuseExistingChunk: true
          },
          
          // React ecosystem
          react: {
            test: /[\\/]node_modules[\\/](react|react-dom|react-router)[\\/]/,
            name: 'react',
            priority: 20,
            reuseExistingChunk: true
          },
          
          // UI Libraries
          ui: {
            test: /[\\/]node_modules[\\/](@radix-ui|lucide-react|framer-motion)[\\/]/,
            name: 'ui',
            priority: 15,
            reuseExistingChunk: true
          },
          
          // Charts et visualisations
          charts: {
            test: /[\\/]node_modules[\\/](recharts|d3)[\\/]/,
            name: 'charts',
            priority: 15,
            reuseExistingChunk: true
          },
          
          // Utilitaires
          utils: {
            test: /[\\/]node_modules[\\/](lodash|date-fns|uuid)[\\/]/,
            name: 'utils',
            priority: 12,
            reuseExistingChunk: true
          },
          
          // Common components
          common: {
            test: /[\\/]src[\\/]components[\\/]/,
            name: 'common',
            priority: 8,
            minChunks: 2,
            reuseExistingChunk: true
          }
        }
      },
      
      // Runtime chunk séparé
      runtimeChunk: {
        name: 'runtime'
      }
    },
    
    // Résolution optimisée
    resolve: {
      alias: {
        // Optimisations d'imports
        'lodash': 'lodash-es'
      },
      extensions: ['.js', '.jsx', '.ts', '.tsx', '.json'],
      modules: ['node_modules'],
      symlinks: false
    },
    
    // Module rules optimisés
    module: {
      rules: [
        // TypeScript/JavaScript
        {
          test: /\.(ts|tsx|js|jsx)$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'babel-loader',
              options: {
                presets: [
                  ['@babel/preset-env', {
                    useBuiltIns: 'usage',
                    corejs: 3,
                    modules: false
                  }],
                  '@babel/preset-react',
                  '@babel/preset-typescript'
                ],
                plugins: [
                  // Optimisations
                  ['babel-plugin-transform-imports', {
                    'lodash': {
                      transform: 'lodash/${member}',
                      preventFullImport: true
                    },
                    'date-fns': {
                      transform: 'date-fns/${member}',
                      preventFullImport: true
                    }
                  }],
                  
                  // Tree shaking pour les imports
                  'babel-plugin-import',
                  
                  // Optimisation des re-renders React
                  'babel-plugin-transform-react-remove-prop-types'
                ]
              }
            }
          ]
        },
        
        // CSS optimisé
        {
          test: /\.css$/,
          use: [
            config.environment === 'production' ? 'mini-css-extract-plugin' : 'style-loader',
            {
              loader: 'css-loader',
              options: {
                importLoaders: 1,
                modules: {
                  auto: true,
                  localIdentName: config.environment === 'production' 
                    ? '[hash:base64:5]' 
                    : '[name]__[local]--[hash:base64:5]'
                }
              }
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [
                    'tailwindcss',
                    'autoprefixer',
                    ...(config.environment === 'production' ? [
                      'cssnano',
                      'postcss-combine-duplicated-selectors'
                    ] : [])
                  ]
                }
              }
            }
          ]
        },
        
        // Images optimisées
        {
          test: /\.(png|jpg|jpeg|gif|svg|webp|avif)$/,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024 // 8KB
            }
          },
          generator: {
            filename: 'assets/images/[name].[hash:8][ext]'
          },
          use: [
            {
              loader: 'image-webpack-loader',
              options: {
                mozjpeg: {
                  progressive: true,
                  quality: 85
                },
                optipng: {
                  enabled: false
                },
                pngquant: {
                  quality: [0.65, 0.90],
                  speed: 4
                },
                gifsicle: {
                  interlaced: false
                },
                webp: {
                  quality: 85
                }
              }
            }
          ]
        }
      ]
    },
    
    // Plugins optimisés
    plugins: [
      // Bundle analyzer en développement
      ...(config.bundleAnalysis ? [{
        plugin: 'webpack-bundle-analyzer',
        options: {
          analyzerMode: 'static',
          openAnalyzer: false,
          reportFilename: 'bundle-report.html'
        }
      }] : []),
      
      // Compression
      ...(config.compression ? [{
        plugin: 'compression-webpack-plugin',
        options: {
          algorithm: 'gzip',
          test: /\.(js|css|html|svg)$/,
          threshold: 8192,
          minRatio: 0.8
        }
      }] : []),
      
      // PWA
      ...(config.progressiveWebApp ? [{
        plugin: 'workbox-webpack-plugin',
        options: {
          generateSW: true,
          clientsClaim: true,
          skipWaiting: true,
          runtimeCaching: [
            {
              urlPattern: /^https:\/\/fonts\.googleapis\.com/,
              handler: 'StaleWhileRevalidate',
              options: {
                cacheName: 'google-fonts-stylesheets'
              }
            },
            {
              urlPattern: /^https:\/\/fonts\.gstatic\.com/,
              handler: 'CacheFirst',
              options: {
                cacheName: 'google-fonts-webfonts',
                expiration: {
                  maxEntries: 30,
                  maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
                }
              }
            }
          ]
        }
      }] : [])
    ],
    
    // Performance hints
    performance: {
      maxEntrypointSize: PERFORMANCE_BUDGET.maxEntrySize,
      maxAssetSize: PERFORMANCE_BUDGET.maxAssetSize,
      hints: config.environment === 'production' ? 'warning' : false
    }
  };
};

// Analyse des bundles
export class BundleAnalyzer {
  static analyze(webpackStats: any): BundleAnalysisResult {
    const assets = webpackStats.toJson().assets;
    const chunks = webpackStats.toJson().chunks;
    
    const totalSize = assets.reduce((total: number, asset: any) => total + asset.size, 0);
    
    // Estimation de la taille gzippée (environ 30% de réduction)
    const gzippedSize = Math.round(totalSize * 0.7);
    
    const chunkAnalysis = chunks.map((chunk: any) => ({
      name: chunk.names[0] || 'unnamed',
      size: chunk.size,
      modules: chunk.modules ? chunk.modules.length : 0
    }));
    
    const assetAnalysis = assets.map((asset: any) => ({
      name: asset.name,
      size: asset.size,
      type: this.getAssetType(asset.name)
    }));
    
    const dependencies = this.analyzeDependencies(webpackStats);
    
    return {
      totalSize,
      gzippedSize,
      chunks: chunkAnalysis,
      assets: assetAnalysis,
      dependencies
    };
  }
  
  private static getAssetType(filename: string): string {
    if (filename.endsWith('.js')) return 'javascript';
    if (filename.endsWith('.css')) return 'stylesheet';
    if (filename.match(/\.(png|jpg|jpeg|gif|svg|webp|avif)$/)) return 'image';
    if (filename.match(/\.(woff|woff2|ttf|eot)$/)) return 'font';
    return 'other';
  }
  
  private static analyzeDependencies(webpackStats: any): Record<string, number> {
    const modules = webpackStats.toJson().modules || [];
    const dependencies: Record<string, number> = {};
    
    modules.forEach((module: any) => {
      if (module.name && module.name.includes('node_modules')) {
        const match = module.name.match(/node_modules\/([@\w][^\/]*)/);
        if (match) {
          const depName = match[1];
          dependencies[depName] = (dependencies[depName] || 0) + (module.size || 0);
        }
      }
    });
    
    return dependencies;
  }
  
  static checkPerformanceBudget(analysis: BundleAnalysisResult, budget: PerformanceBudget): Array<{
    rule: string;
    actual: number;
    budget: number;
    passed: boolean;
  }> {
    const results = [];
    
    // Vérifier la taille totale initiale
    const initialChunks = analysis.chunks.filter(chunk => 
      chunk.name.includes('main') || chunk.name.includes('vendor') || chunk.name.includes('runtime')
    );
    const initialSize = initialChunks.reduce((total, chunk) => total + chunk.size, 0);
    
    results.push({
      rule: 'Initial Bundle Size',
      actual: initialSize,
      budget: budget.maxInitialBundleSize,
      passed: initialSize <= budget.maxInitialBundleSize
    });
    
    // Vérifier les chunks individuels
    analysis.chunks.forEach(chunk => {
      results.push({
        rule: `Chunk Size (${chunk.name})`,
        actual: chunk.size,
        budget: budget.maxChunkSize,
        passed: chunk.size <= budget.maxChunkSize
      });
    });
    
    // Vérifier les assets
    analysis.assets.forEach(asset => {
      if (asset.type === 'javascript' || asset.type === 'stylesheet') {
        results.push({
          rule: `Asset Size (${asset.name})`,
          actual: asset.size,
          budget: budget.maxAssetSize,
          passed: asset.size <= budget.maxAssetSize
        });
      }
    });
    
    return results;
  }
}

// Optimisations spécifiques aux assets
export class AssetOptimizer {
  static optimizeImages() {
    return {
      // Configuration pour image-webpack-loader
      mozjpeg: {
        progressive: true,
        quality: 85
      },
      optipng: {
        enabled: false // Utiliser pngquant à la place
      },
      pngquant: {
        quality: [0.65, 0.90],
        speed: 4
      },
      gifsicle: {
        interlaced: false
      },
      webp: {
        quality: 85
      },
      svgo: {
        plugins: [
          { removeViewBox: false },
          { removeDimensions: true }
        ]
      }
    };
  }
  
  static optimizeFonts() {
    return {
      // Préchargement des polices critiques
      preload: [
        '/fonts/inter-var.woff2'
      ],
      
      // Stratégies de chargement
      fontDisplay: 'swap',
      
      // Subset des polices
      subset: 'latin',
      
      // Compression
      woff2: true
    };
  }
  
  static optimizeCSS() {
    return {
      // PurgeCSS pour supprimer le CSS inutilisé
      purge: {
        content: ['./src/**/*.{js,jsx,ts,tsx}'],
        defaultExtractor: (content: string) => content.match(/[\w-/:]+(?<!:)/g) || []
      },
      
      // Optimisations PostCSS
      postcss: [
        'autoprefixer',
        'cssnano',
        'postcss-combine-duplicated-selectors',
        'postcss-merge-rules'
      ]
    };
  }
}

// Configuration pour le service worker
export const getServiceWorkerConfig = () => ({
  // Stratégies de cache
  runtimeCaching: [
    // API calls
    {
      urlPattern: /^https:\/\/.*\/api\//,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        networkTimeoutSeconds: 3,
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 5 * 60 // 5 minutes
        }
      }
    },
    
    // Static assets
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'images',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
        }
      }
    },
    
    // Fonts
    {
      urlPattern: /\.(?:woff|woff2|ttf|eot)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'fonts',
        expiration: {
          maxEntries: 30,
          maxAgeSeconds: 365 * 24 * 60 * 60 // 1 year
        }
      }
    }
  ],
  
  // Fichiers à précacher
  precacheManifest: [
    '/index.html',
    '/static/css/main.css',
    '/static/js/main.js'
  ],
  
  // Skip waiting pour les mises à jour
  skipWaiting: true,
  clientsClaim: true
});

// Générateur de manifeste PWA
export const generatePWAManifest = () => ({
  name: 'M&A Intelligence Platform',
  short_name: 'M&A Intelligence',
  description: 'Plateforme d\'intelligence M&A pour analystes financiers',
  start_url: '/',
  display: 'standalone',
  theme_color: '#3B82F6',
  background_color: '#FFFFFF',
  orientation: 'portrait-primary',
  icons: [
    {
      src: '/icons/icon-192x192.png',
      sizes: '192x192',
      type: 'image/png'
    },
    {
      src: '/icons/icon-512x512.png',
      sizes: '512x512',
      type: 'image/png'
    },
    {
      src: '/icons/icon-512x512.png',
      sizes: '512x512',
      type: 'image/png',
      purpose: 'maskable'
    }
  ],
  categories: ['business', 'finance', 'productivity'],
  lang: 'fr',
  scope: '/'
});

export default {
  PRODUCTION_BUILD_CONFIG,
  PERFORMANCE_BUDGET,
  getOptimizedWebpackConfig,
  BundleAnalyzer,
  AssetOptimizer,
  getServiceWorkerConfig,
  generatePWAManifest
};