import { useState, useEffect } from 'react'
import { LayoutDashboard, Inbox, Users, BarChart2, Home, BrainCircuit } from 'lucide-react'
import OverviewTab    from './tabs/OverviewTab'
import InboxTab       from './tabs/InboxTab'
import CandidatesTab  from './tabs/CandidatesTab'
import PerformanceTab from './tabs/PerformanceTab'
import AnalyseTab     from './tabs/AnalyseTab'

const TABS = [
  { id: 'overview',    label: 'Vue d\'ensemble', icon: Home },
  { id: 'inbox',       label: 'Inbox',           icon: Inbox },
  { id: 'candidates',  label: 'Candidats',        icon: Users },
  { id: 'analyse',     label: 'Analyse IA',       icon: BrainCircuit },
  { id: 'performance', label: 'Performance',      icon: BarChart2 },
]

export default function App() {
  const [tab, setTab]               = useState('overview')
  const [candidates, setCandidates] = useState([])
  const [stats, setStats]           = useState(null)

  const refresh = () => {
    fetch('/candidates').then(r => r.json()).then(setCandidates).catch(() => {})
    fetch('/stats').then(r => r.json()).then(setStats).catch(() => {})
  }

  useEffect(() => { refresh() }, [])

  const inboxCount = candidates.filter(c => !c.status || c.status === 'inbox').length

  return (
    <div className="min-h-screen flex flex-col">

      {/* Header */}
      <header className="glass-strong px-8 py-3.5 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center shadow-sm">
            <LayoutDashboard size={15} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-800 leading-none tracking-tight">CV-Intelligence</h1>
            <p className="text-slate-400 text-xs mt-0.5">LuxTalent Advisory Group · Dashboard RH</p>
          </div>
        </div>

        {stats && (
          <div className="flex gap-5">
            <Stat label="Total"       value={stats.total} />
            <Stat label="Invités"     value={stats.invited}  color="text-emerald-600" />
            <Stat label="Rejetés"     value={stats.rejected} color="text-red-500" />
            <Stat label="Taux"        value={`${stats.invite_rate}%`} color="text-blue-600" />
            <Stat label="Aujourd'hui" value={stats.today}    color="text-slate-600" />
          </div>
        )}
      </header>

      {/* Tab bar */}
      <nav className="bg-white/80 backdrop-blur border-b border-slate-200/80 px-6 flex gap-0.5 sticky top-[57px] z-30">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            className={`relative flex items-center gap-2 px-4 py-3 text-sm font-medium transition-all rounded-t-lg
              ${tab === id ? 'tab-active' : 'tab-inactive'}`}>
            <Icon size={14} />
            {label}
            {id === 'inbox' && inboxCount > 0 && (
              <span className="absolute -top-0.5 right-0.5 min-w-[17px] h-[17px] bg-blue-600 text-white text-[9px] font-bold rounded-full flex items-center justify-center px-1 shadow">
                {inboxCount}
              </span>
            )}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="flex-1 p-6 overflow-auto">
        {tab === 'overview'    && <OverviewTab    candidates={candidates} />}
        {tab === 'inbox'       && <InboxTab       candidates={candidates} onUpdate={refresh} />}
        {tab === 'candidates'  && <CandidatesTab  candidates={candidates} />}
        {tab === 'analyse'     && <AnalyseTab     candidates={candidates} />}
        {tab === 'performance' && <PerformanceTab />}
      </main>
    </div>
  )
}

function Stat({ label, value, color = 'text-slate-700' }) {
  return (
    <div className="text-center">
      <div className={`text-base font-bold leading-none ${color}`}>{value ?? '—'}</div>
      <div className="text-slate-400 text-xs mt-0.5">{label}</div>
    </div>
  )
}
