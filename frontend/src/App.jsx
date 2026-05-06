import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import TopNav from './components/TopNav'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import CandidateList from './pages/CandidateList'
import CandidateProfile from './pages/CandidateProfile'
import Calendar from './pages/Calendar'
import Settings from './pages/Settings'

export default function App() {
  const [page, setPage]               = useState('dashboard')
  const [candidateId, setCandidateId] = useState(null)

  const PAGES = ['dashboard', 'jobs', 'candidates', 'calendar', 'settings']

  const navigate = (target, id = null) => {
    if (id) setCandidateId(id)
    setPage(target)
  }

  // Alt + flèches pour naviguer entre les pages
  useEffect(() => {
    const handler = (e) => {
      if (!e.altKey) return
      const idx = PAGES.indexOf(page === 'profile' ? 'candidates' : page)
      if (e.key === 'ArrowRight' && idx < PAGES.length - 1) navigate(PAGES[idx + 1])
      if (e.key === 'ArrowLeft'  && idx > 0)                navigate(PAGES[idx - 1])
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
