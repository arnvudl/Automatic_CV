import { useState } from 'react'
import Sidebar from './components/Sidebar'
import TopNav from './components/TopNav'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import CandidateList from './pages/CandidateList'
import CandidateProfile from './pages/CandidateProfile'
import Calendar from './pages/Calendar'

export default function App() {
  const [page, setPage] = useState('dashboard')

  // Map sidebar nav ids to pages
  // 'candidates' opens the list; 'profile' is navigated from within
  const navigate = (target) => setPage(target)

  const renderPage = () => {
    switch (page) {
      case 'dashboard':  return <Dashboard onNavigate={navigate} />
      case 'jobs':       return <Jobs />
      case 'candidates': return <CandidateList onNavigate={navigate} />
      case 'profile':    return <CandidateProfile onNavigate={navigate} />
      case 'calendar':   return <Calendar />
      case 'settings':   return <PlaceholderPage title="Settings" icon="settings" />
      default:           return <Dashboard onNavigate={navigate} />
    }
  }

  // The sidebar "active" maps profile → candidates to keep nav highlighted
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

function PlaceholderPage({ title, icon }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-on-surface-variant">
      <span className="material-symbols-outlined" style={{ fontSize: 64 }}>{icon}</span>
      <h2 className="text-2xl font-bold text-on-surface">{title}</h2>
      <p className="text-sm">Cette page arrive bientôt.</p>
    </div>
  )
}
