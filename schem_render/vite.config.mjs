import { defineConfig } from 'vite';
export default defineConfig({
  base: '/',
  build: {
    outDir: 'dist_app',
    assetsInlineLimit: 0,
    target: 'es2022',
    rollupOptions: {
      output: { manualChunks: undefined },
    },
  },
  optimizeDeps: { exclude: [] },
  worker: { format: 'es' },
});
