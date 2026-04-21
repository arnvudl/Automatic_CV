import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/candidates': 'http://localhost:8000',
      '/stats':      'http://localhost:8000',
      '/score':      'http://localhost:8000',
      '/comments':   'http://localhost:8000',
      '/analyse':    'http://localhost:8000',
    }
  }
})
