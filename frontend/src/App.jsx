import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import TopNav from './components/TopNav'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import CandidateList from './pages/CandidateList'
import CandidateProfile from './pages/CandidateProfile'
import Calendar from './pages/Calendar'
import Settings from './pages/Settings'
import Archives from './pages/Archives'

// ── URL ↔ page mapping ───────────────────────────────────────────────
const ROUTE_MAP = {
  '/':            'dashboard',
  '/dashboard':   'dashboard',
  '/jobs':        'jobs',
  '/candidates':  'candidates',
  '/calendar':    'calendar',
  '/settings':    'settings',
  '/archives':    'archives',
}

const PAGE_TO_PATH = {
  dashboard:  '/dashboard',
  jobs:       '/jobs',
  candidates: '/candidates',
  profile:    '/candidates',   // profile garde l'URL /candidates
  calendar:   '/calendar',
  settings:   '/settings',
  archives:   '/archives',
}

const NAV_PAGES = ['dashboard', 'jobs', 'candidates', 'archives', 'calendar', 'settings']

function getPageFromPath(path) {
  return ROUTE_MAP[path] ?? 'dashboard'
}

export default function App() {
  const [page, setPage]               = useState(() => getPageFromPath(window.location.pathname))
  const [candidateId, setCandidateId] = useState(null)

  const navigate = (target, id = null) => {
    if (id) setCandidateId(id)
    const path = PAGE_TO_PATH[target] ?? '/dashboard'
    window.history.pushState({ page: target, id }, '', path)
    setPage(target)
  }

  // Gérer le bouton retour du navigateur
  useEffect(() => {
    const handler = (e) => {
      const state = e.state
      if (state?.page) {
        if (state.id) setCandidateId(state.id)
        setPage(state.page)
      } else {
        setPage(getPageFromPath(window.location.pathname))
      }
    }
    window.addEventListener('popstate', handler)
    return () => window.removeEventListener('popstate', handler)
  }, [])

  // Alt + flèches pour naviguer entre les pages
  useEffect(() => {
    const handler = (e) => {
      if (!e.altKey) return
      const current = page === 'profile' ? 'candidates' : page
      const idx = NAV_PAGES.indexOf(current)
      if (e.key === 'ArrowRight' && idx < NAV_PAGES.length - 1) navigate(NAV_PAGES[idx + 1])
      if (e.key === 'ArrowLeft'  && idx > 0)                    navigate(NAV_PAGES[idx - 1])
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [page])

  const renderPage = () => {
    switch (page) {
      case 'dashboard':  return <Dashboard onNavigate={navigate} />
      case 'jobs':       return <Jobs onNavigate={navigate} />
      case 'candidates': return <CandidateList onNavigate={navigate} />
      case 'profile':    return <CandidateProfile onNavigate={navigate} candidateId={candidateId} />
      case 'calendar':   return <Calendar onNavigate={navigate} />
      case 'settings':   return <Settings onNavigate={navigate} />
      case 'archives':   return <Archives onNavigate={navigate} />
      default:           return <Dashboard onNavigate={navigate} />
    }
  }

  const sidebarActive = page === 'profile' ? 'candidates' : page

  return (
    <div className="bg-surface text-on-surface antialiased min-h-screen">
      <Sidebar active={sidebarActive} onNavigate={navigate} />
      <div className="ml-64 min-h-screen flex flex-col">
        <TopNav onNavigate={navigate} />
        <div className="flex-1">
          {renderPage()}
        </div>
      </div>
    </div>
  )
}
