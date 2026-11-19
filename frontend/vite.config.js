import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Development: Vite on :5173 proxies /api and /docs to the FastAPI server on :8000 (python -m api).
// Production: `npm run build` writes dist/, which the FastAPI server serves itself.
const API = process.env.PLANKTOSCAN_API || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': API, '/docs': API, '/openapi.json': API },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
  },
});
