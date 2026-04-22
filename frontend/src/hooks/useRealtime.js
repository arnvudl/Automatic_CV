/**
 * useRealtime — connexion SSE au backend FastAPI
 *
 * Reconnexion automatique toutes les 3s si la connexion est perdue.
 * Les handlers sont des callbacks { onCandidateScored, onStatusUpdated, onConnected }.
 */

import { useEffect, useRef, useCallback } from 'react'

const BASE = import.meta.env.VITE_API_URL ?? ''
const SSE_URL = `${BASE}/events`

export function useRealtime({ onCandidateScored, onStatusUpdated, onConnected } = {}) {
  const esRef    = useRef(null)
  const retryRef = useRef(null)

  const connect = useCallback(() => {
    if (esRef.current) esRef.current.close()

    const es = new EventSource(SSE_URL)
    esRef.current = es

    es.addEventListener('connected', () => {
      onConnected?.()
    })

    es.addEventListener('candidate_scored', (e) => {
      try { onCandidateScored?.(JSON.parse(e.data)) } catch (_) {}
    })

    es.addEventListener('status_updated', (e) => {
      try { onStatusUpdated?.(JSON.parse(e.data)) } catch (_) {}
    })

    es.onerror = () => {
      es.close()
      esRef.current = null
      // Reconnexion automatique après 3s
      retryRef.current = setTimeout(connect, 3000)
    }
  }, [onCandidateScored, onStatusUpdated, onConnected])

  useEffect(() => {
    connect()
    return () => {
      esRef.current?.close()
      clearTimeout(retryRef.current)
    }
  }, [connect])
}
