/**
 * api.js — Client HTTP centralisé
 *
 * En local  : VITE_API_URL vide → proxy Vite → localhost:8000
 * Sur Vercel : VITE_API_URL=https://api.lony.app
 */

const BASE      = import.meta.env.VITE_API_URL ?? ''
const TOKEN_KEY = 'lony_token'

function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

async function request(path, options = {}) {
  const token = getToken()
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
    ...options,
  })

  // Token expiré ou invalide → déconnexion
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY)
    window.dispatchEvent(new Event('lony:logout'))
    throw new Error('Session expirée')
  }

  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${err}`)
  }
  return res.json()
}

// ── Auth ─────────────────────────────────────────────────────────────
export const loginUser = (email, password) =>
  fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  }).then(r => r.json())

// ── Candidates ───────────────────────────────────────────────────────
export const getCandidates = (params = {}) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString()
  return request(`/candidates${qs ? '?' + qs : ''}`)
}

export const getCandidate  = (id) => request(`/candidates/${id}`)

export const updateStatus  = (id, status) =>
  request(`/candidates/${id}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  })

export const deleteCandidate = (id) =>
  request(`/candidates/${id}`, { method: 'DELETE' })

// ── Stats ────────────────────────────────────────────────────────────
export const getStats = () => request('/stats')

// ── Score (upload CV) ────────────────────────────────────────────────
export const scoreCV = (file) => {
  const fd = new FormData()
  fd.append('file', file)
  const token = getToken()
  return fetch(`${BASE}/score`, {
    method: 'POST',
    body: fd,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  }).then(r => r.json())
}

// ── Comments ─────────────────────────────────────────────────────────
export const getComments   = (id)             => request(`/comments/${id}`)
export const addComment    = (id, author, text) =>
  request(`/comments/${id}`, { method: 'POST', body: JSON.stringify({ author, text }) })
export const deleteComment = (id, cid)        =>
  request(`/comments/${id}/${cid}`, { method: 'DELETE' })

// ── Jobs ─────────────────────────────────────────────────────────────
export const getJobs   = (params = {}) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString()
  return request(`/jobs${qs ? '?' + qs : ''}`)
}
export const createJob = (data)    => request('/jobs', { method: 'POST',  body: JSON.stringify(data) })
export const updateJob = (id, data) => request(`/jobs/${id}`, { method: 'PATCH', body: JSON.stringify(data) })
export const deleteJob = (id)      => request(`/jobs/${id}`, { method: 'DELETE' })

// ── Interviews ───────────────────────────────────────────────────────
export const getInterviews = (params = {}) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString()
  return request(`/interviews${qs ? '?' + qs : ''}`)
}
export const createInterview = (data) =>
  request('/interviews', { method: 'POST', body: JSON.stringify(data) })
export const updateInterview = (id, data) =>
  request(`/interviews/${id}`, { method: 'PATCH', body: JSON.stringify(data) })
export const deleteInterview = (id) =>
  request(`/interviews/${id}`, { method: 'DELETE' })

// ── Scorecards ───────────────────────────────────────────────────────
export const getExplain       = (candidateId) => request(`/candidates/${candidateId}/explain`)
export const getScorecards    = (candidateId) => request(`/candidates/${candidateId}/scorecards`)
export const createScorecard  = (candidateId, data) =>
  request(`/candidates/${candidateId}/scorecards`, { method: 'POST', body: JSON.stringify(data) })
export const deleteScorecard  = (scorecardId) => request(`/scorecards/${scorecardId}`, { method: 'DELETE' })

// ── Pipeline Kanban ──────────────────────────────────────────────────
export const getPipelineStages = (jobId) =>
  request(`/jobs/${jobId}/stages`)

export const createPipelineStage = (jobId, data) =>
  request(`/jobs/${jobId}/stages`, { method: 'POST', body: JSON.stringify(data) })

export const moveCandidateStage = (candidateId, stageId) =>
  request(`/candidates/${candidateId}/stage`, {
    method: 'PATCH',
    body: JSON.stringify({ stage_id: stageId }),
  })
