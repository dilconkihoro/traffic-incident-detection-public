import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// Dev-only proxy: keeps the frontend free of any hardcoded backend URL and avoids CORS
// entirely in development. The production build is served by (or alongside) the backend
// directly, so no proxy or absolute URL is ever needed there either.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
