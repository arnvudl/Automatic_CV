/**
 * api.js — Client HTTP centralisé
 *
 * En local  : VITE_API_URL est vide → même origine (proxy Vite → localhost:8000)
 * Sur Vercel : VITE_API_URL=https://api.votredomaine.com
 */

const BASE = import.meta.env.VITE_API_URL ?? ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${err}`)
  }
  return res.json()
}

// ── Candidates ───────────────────────────────────────────────────────
export const getCandidates = (params = {}) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString()
  return request(`/candidates${qs ? '?' + qs : ''}`)
}

export const getCandidate = (id) => request(`/candidates/${id}`)

export const updateStatus = (id, status) =>
  request(`/candidates/${id}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  })

// ── Stats ────────────────────────────────────────────────────────────
export const getStats = () => request('/stats')

// ── Score (upload CV) ────────────────────────────────────────────────
export const scoreCV = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  return fetch(`${BASE}/score`, { method: 'POST', body: fd }).then((r) => r.json())
}

// ── Comments ─────────────────────────────────────────────────────────
export const getComments    = (id)           => request(`/comments/${id}`)
export const addComment     = (id, author, text) =>
  request(`/comments/${id}`, { method: 'POST', body: JSON.stringify({ author, text }) })
export const deleteComment  = (id, cid)      =>
  request(`/comments/${id}/${cid}`, { method: 'DELETE' })
