import { useState } from 'react'
import { Icon } from '../components/Icon'

const CANDIDATES = [
  { name: 'Elena Smith', role: 'Senior Engineer @ TechFlow', score: 92, date: 'Oct 12, 2023', stage: 'Interviewing', stageColor: 'bg-primary/10 text-primary', dot: 'bg-primary', initials: 'ES', color: 'bg-blue-100 text-primary' },
  { name: 'Marcus Chen', role: 'Lead Frontend @ CloudScale', score: 75, date: 'Oct 14, 2023', stage: 'Screening', stageColor: 'bg-surface-container-highest text-on-surface-variant', dot: 'bg-outline', initials: 'MC', color: 'bg-secondary-fixed text-secondary' },
  { name: 'Jordan Hayes', role: 'Frontend Developer @ StartupX', score: 85, date: 'Oct 15, 2023', stage: 'Sourcing', stageColor: 'bg-surface-container-highest text-on-surface-variant', dot: 'bg-outline', initials: 'JH', color: 'bg-green-100 text-tertiary' },
  { name: 'Aisha Mohammed', role: 'UX/UI Developer @ GlobalNet', score: 95, date: 'Oct 16, 2023', stage: 'Interviewing', stageColor: 'bg-primary/10 text-primary', dot: 'bg-primary', initials: 'AM', color: 'bg-purple-100 text-purple-700' },
]

function ScoreRing({ score }) {
  const color = score >= 85 ? '#006b2b' : score >= 70 ? '#ca8a04' : '#ba1a1a'
  const labelColor = score >= 85 ? 'text-tertiary' : score >= 70 ? 'text-yellow-600' : 'text-error'
  const badgeBg = score >= 85 ? 'bg-tertiary/10 text-tertiary' : 'bg-yellow-100 text-yellow-700'
  const badgeLabel = score >= 85 ? 'TOP MATCH' : 'GOOD MATCH'
  const offset = 125.6 * (1 - score / 100)
  return (
    <div className="flex items-center gap-3">
      <div className="w-12 h-12 rounded-full flex items-center justify-center relative">
        <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r="20" fill="none" stroke={`${color}33`} strokeWidth="4" />
          <circle cx="24" cy="24" r="20" fill="none" stroke={color} strokeWidth="4"
            strokeDasharray="125.6" strokeDashoffset={offset} strokeLinecap="round" />
        </svg>
        <span className={`text-xs font-black ${labelColor}`}>{score}%</span>
      </div>
      <span className={`px-2 py-1 text-[10px] font-bold rounded ${badgeBg}`}>{badgeLabel}</span>
    </div>
  )
}

export default function CandidateList({ onNavigate }) {
  const [scoreFilter, setScoreFilter] = useState('all')

  const filtered = CANDIDATES.filter(c => {
    if (scoreFilter === 'high') return c.score >= 80
    if (scoreFilter === 'mid') return c.score >= 50 && c.score < 80
    if (scoreFilter === 'low') return c.score < 50
    return true
  })

  return (
    <div className="flex min-h-screen">
      {/* Sidebar Filters */}
      <aside className="w-72 flex-shrink-0 p-8 space-y-8">
        <div className="bg-surface-container-low p-6 rounded-3xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-on-surface">Filters</h3>
            <button className="text-primary text-sm font-semibold" onClick={() => setScoreFilter('all')}>Clear all</button>
          </div>

          {/* Skills */}
          <div className="mb-8">
            <label className="block text-sm font-bold text-on-surface mb-3">Skills Required</label>
            <div className="space-y-2">
              {['React.js', 'TypeScript', 'Tailwind CSS', 'Next.js'].map((skill, i) => (
                <label key={skill} className="flex items-center gap-3 cursor-pointer group">
                  <input type="checkbox" defaultChecked={i < 2}
                    className="rounded border-outline-variant text-primary focus:ring-primary/20 w-5 h-5 accent-primary" />
                  <span className="text-sm text-on-surface-variant group-hover:text-on-surface">{skill}</span>
                </label>
              ))}
            </div>
          </div>

          {/* AI Score filter */}
          <div>
            <label className="block text-sm font-bold text-on-surface mb-3">AI Match Score</label>
            <div className="space-y-3">
              {[
                { id: 'high', label: 'High Match (>80%)', count: 3, active: 'bg-white text-tertiary border border-tertiary/10' },
                { id: 'mid', label: 'Mid Match (50-80%)', count: 1, active: 'bg-white text-on-surface-variant' },
                { id: 'low', label: 'Needs Review (<50%)', count: 0, active: 'bg-white text-on-surface-variant' },
              ].map(({ id, label, count, active }) => (
                <button key={id}
                  onClick={() => setScoreFilter(scoreFilter === id ? 'all' : id)}
                  className={`w-full text-left px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center justify-between
                    ${scoreFilter === id ? active : 'text-on-surface-variant hover:bg-white'}`}>
                  <span>{label}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] ${scoreFilter === id ? 'bg-tertiary/10' : 'bg-surface-container-highest'}`}>{count}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* AI Insights */}
        <div className="bg-gradient-to-br from-tertiary to-tertiary-container p-6 rounded-3xl text-on-tertiary shadow-lg relative overflow-hidden">
          <div className="relative z-10">
            <Icon name="auto_awesome" fill size={36} />
            <h4 className="text-lg font-bold mt-4 mb-2">Talent Pulse AI</h4>
            <p className="text-sm opacity-90 leading-relaxed">We found 3 candidates with perfect tech stack alignment for this role.</p>
            <button className="mt-4 px-4 py-2 bg-white/20 hover:bg-white/30 backdrop-blur-md rounded-xl text-xs font-bold transition-all">
              View Insights
            </button>
          </div>
          <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-white/10 rounded-full blur-2xl" />
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 p-8 pl-0">
        {/* Header */}
        <div className="mb-10 flex justify-between items-end">
          <div>
            <nav className="flex items-center gap-2 text-xs text-on-surface-variant mb-2">
              <span>Jobs</span>
              <Icon name="chevron_right" size={14} />
              <span>Active Postings</span>
            </nav>
            <h1 className="text-6xl font-black text-on-surface leading-tight tracking-tight">
              Senior Frontend<br />Developer
            </h1>
            <div className="flex items-center gap-4 mt-4">
              <span className="px-3 py-1 bg-secondary-container/30 text-on-secondary-container rounded-full text-sm font-medium">Remote</span>
              <span className="px-3 py-1 bg-secondary-container/30 text-on-secondary-container rounded-full text-sm font-medium">Full-time</span>
              <span className="text-on-surface-variant text-sm font-medium flex items-center gap-1">
                <Icon name="group" size={16} /> 24 Candidates Applied
              </span>
            </div>
          </div>
          <div className="flex gap-3">
            <button className="px-5 py-2.5 bg-surface-container-high text-on-surface rounded-full font-medium hover:bg-surface-container-highest transition-all flex items-center gap-2 text-sm">
              <Icon name="share" size={18} /> Share
            </button>
            <button className="px-5 py-2.5 bg-surface-container-high text-on-surface rounded-full font-medium hover:bg-surface-container-highest transition-all flex items-center gap-2 text-sm">
              <Icon name="edit" size={18} /> Edit
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="bg-surface-container-lowest rounded-[2rem] overflow-hidden shadow-ambient">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-container-low/50">
                {['Candidate Name', 'AI Match Score', 'Applied Date', 'Current Stage', ''].map(h => (
                  <th key={h} className="px-6 py-5 text-[11px] font-bold text-outline uppercase tracking-wider first:pl-8 last:pr-8 last:text-right">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((c, i) => (
                <tr key={i}
                  className="hover:bg-surface-container-low/30 transition-colors group cursor-pointer"
                  onClick={() => onNavigate('profile')}>
                  <td className="px-8 py-5">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-2xl ${c.color} flex items-center justify-center font-bold flex-shrink-0`}>
                        {c.initials}
                      </div>
                      <div>
                        <div className="font-bold text-on-surface">{c.name}</div>
                        <div className="text-xs text-on-surface-variant">{c.role}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-5">
                    <ScoreRing score={c.score} />
                  </td>
                  <td className="px-6 py-5 text-sm text-on-surface-variant font-medium">{c.date}</td>
                  <td className="px-6 py-5">
                    <span className={`px-4 py-1.5 ${c.stageColor} rounded-full text-xs font-bold flex items-center gap-2 w-fit`}>
                      <span className={`w-1.5 h-1.5 ${c.dot} rounded-full`} />
                      {c.stage}
                    </span>
                  </td>
                  <td className="px-8 py-5 text-right">
                    <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant"><Icon name="chat_bubble" size={18} /></button>
                      <button className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant"><Icon name="visibility" size={18} /></button>
                      <button className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant"><Icon name="more_vert" size={18} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div className="px-8 py-6 bg-surface-container-low/30 flex justify-between items-center">
            <span className="text-xs font-bold text-outline uppercase tracking-widest">Showing 1 to {filtered.length} of {CANDIDATES.length} candidates</span>
            <div className="flex gap-2">
              {['chevron_left', '1', '2', '3', 'chevron_right'].map((item, i) => (
                <button key={i} className={`w-8 h-8 flex items-center justify-center rounded-lg transition-all text-sm
                  ${item === '1' ? 'bg-white text-primary font-bold shadow-sm' : 'hover:bg-white text-on-surface-variant'}`}>
                  {['chevron_left', 'chevron_right'].includes(item) ? <Icon name={item} size={16} /> : item}
                </button>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* FAB */}
      <button className="fixed bottom-8 right-8 w-14 h-14 bg-primary text-white rounded-2xl shadow-xl hover:shadow-2xl hover:scale-105 active:scale-95 transition-all flex items-center justify-center z-50">
        <Icon name="add" size={32} />
      </button>
    </div>
  )
}
