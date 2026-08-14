import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
  base: './', // Ensures relative asset resolution on GitHub Pages and custom subpaths
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
