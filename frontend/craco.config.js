const path = require('path');

module.exports = {
  webpack: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@/components': path.resolve(__dirname, 'src/components'),
      '@/lib': path.resolve(__dirname, 'src/lib'),
      '@/hooks': path.resolve(__dirname, 'src/hooks'),
      '@/stores': path.resolve(__dirname, 'src/stores'),
      '@/types': path.resolve(__dirname, 'src/types'),
      '@/services': path.resolve(__dirname, 'src/services'),
      '@/features': path.resolve(__dirname, 'src/features'),
    },
  },
  typescript: {
    enableTypeChecking: true,
  },
};