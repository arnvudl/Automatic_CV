import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/candidates': { target: 'http://localhost:8000', changeOrigin: true },
      '/stats':      { target: 'http://localhost:8000', changeOrigin: true },
      '/score':      { target: 'http://localhost:8000', changeOrigin: true },
      '/comments':   { target: 'http://localhost:8000', changeOrigin: true },
      '/events':     { target: 'http://localhost:8000', changeOrigin: true,
                       ws: false },  // SSE — pas WS
    }
  }
})
