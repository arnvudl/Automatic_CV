import { useState, useEffect } from 'react'
import { useAuth } from './contexts/AuthContext'
import Sidebar from './components/Sidebar'
import TopNav from './components/TopNav'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import CandidateList from './pages/CandidateList'
import CandidateProfile from './pages/CandidateProfile'
import Calendar from './pages/Calendar'
import Settings from './pages/Settings'
import Archives from './pages/Archives'
import Pipeline from './pages/Pipeline'
import LoginPage from './pages/LoginPage'
import { Icon } from './components/Icon'

const ROUTE_MAP = {
  '/':           'dashboard',
  '/dashboard':  'dashboard',
  '/jobs':       'jobs',
  '/pipeline':   'pipeline',
  '/candidates': 'candidates',
  '/calendar':   'calendar',
  '/settings':   'settings',
  '/archives':   'archives',
}

const PAGE_TO_PATH = {
  dashboard:  '/dashboard',
  jobs:       '/jobs',
  pipeline:   '/pipeline',
  candidates: '/candidates',
  profile:    '/candidates',
  calendar:   '/calendar',
  settings:   '/settings',
  archives:   '/archives',
}

const NAV_PAGES = ['dashboard', 'jobs', 'pipeline', 'candidates', 'archives', 'calendar', 'settings']

function getPageFromPath(path) {
  return ROUTE_MAP[path] ?? 'dashboard'
}

export default function App() {
  const { isAuth, loading, logout } = useAuth()
  const [page, setPage]               = useState(() => getPageFromPath(window.location.pathname))
  const [candidateId, setCandidateId] = useState(null)

  useEffect(() => {
    const handler = () => logout()
    window.addEventListener('lony:logout', handler)
    return () => window.removeEventListener('lony:logout', handler)
  }, [logout])

  const navigate = (target, id = null) => {
    if (id) setCandidateId(id)
    const path = PAGE_TO_PATH[target] ?? '/dashboard'
    window.history.pushState({ page: target, id }, '', path)
    setPage(target)
  }

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

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-muted-foreground">
          <div className="w-12 h-12 bg-foreground rounded-xl flex items-center justify-center text-primary-foreground animate-pulse">
            <Icon name="auto_awesome" fill size={24} />
          </div>
          <span className="text-sm font-medium">Chargement...</span>
        </div>
      </div>
    )
  }

  if (!isAuth) return <LoginPage />

  const renderPage = () => {
    switch (page) {
      case 'dashboard':  return <Dashboard onNavigate={navigate} />
      case 'jobs':       return <Jobs onNavigate={navigate} />
      case 'pipeline':   return <Pipeline onNavigate={navigate} />
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
    <div className="bg-background text-foreground antialiased min-h-screen">
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
