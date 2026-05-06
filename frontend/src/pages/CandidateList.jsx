import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getCandidates } from '../lib/api'

const AVATAR_COLORS = [
  'bg-blue-100 text-primary',
  'bg-secondary-fixed text-secondary',
  'bg-green-100 text-tertiary',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

function initials(name) {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

function ScoreRing({ score }) {
  const pct = Math.round((score ?? 0) * 100)
  const color = pct >= 75 ? '#006b2b' : pct >= 50 ? '#ca8a04' : '#ba1a1a'
  const labelColor = pct >= 75 ? 'text-tertiary' : pct >= 50 ? 'text-yellow-600' : 'text-error'
  const badgeBg = pct >= 75 ? 'bg-tertiary/10 text-tertiary' : pct >= 50 ? 'bg-yellow-100 text-yellow-700' : 'bg-error-container text-error'
  const badgeLabel = pct >= 75 ? 'TOP MATCH' : pct >= 50 ? 'GOOD MATCH' : 'LOW MATCH'
  const offset = 125.6 * (1 - pct / 100)
  return (
    <div className="flex items-center gap-3">
      <div className="w-12 h-12 rounded-full flex items-center justify-center relative">
        <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r="20" fill="none" stroke={`${color}33`} strokeWidth="4" />
          <circle cx="24" cy="24" r="20" fill="none" stroke={color} strokeWidth="4"
            strokeDasharray="125.6" strokeDashoffset={offset} strokeLinecap="round" />
        </svg>
        <span className={`text-xs font-black ${labelColor}`}>{pct}%</span>
      </div>
      <span className={`px-2 py-1 text-[10px] font-bold rounded ${badgeBg}`}>{badgeLabel}</span>
    </div>
  )
}

function decisionStage(decision) {
  if (decision === 'invite')     return { label: 'Invité',    color: 'bg-primary/10 text-primary', dot: 'bg-primary' }
  if (decision === 'reject')     return { label: 'Rejeté',    color: 'bg-error-container text-error', dot: 'bg-error' }
  if (decision === 'eliminated') return { label: 'Éliminé',   color: 'bg-surface-container-highest text-on-surface-variant', dot: 'bg-outline' }
  return { label: 'En attente', color: 'bg-surface-container-highest text-on-surface-variant', dot: 'bg-outline' }
}

export default function CandidateList({ onNavigate }) {
  const [candidates, setCandidates] = useState([])
  const [loading, setLoading]       = useState(true)
  const [scoreFilter, setScoreFilter] = useState('all')
  const [sectorFilter, setSectorFilter] = useState('all')
  const [decisionFilter, setDecisionFilter] = useState('all')

  useEffect(() => {
    getCandidates({ limit: 100 })
      .then(setCandidates)
      .catch(() => setCandidates([]))
      .finally(() => setLoading(false))
  }, [])

  // Sectors uniques depuis les données
  const sectors = [...new Set(candidates.map(c => c.sector).filter(Boolean))]

  const filtered = candidates.filter(c => {
    const pct = Math.round((c.score ?? 0) * 100)
    if (scoreFilter === 'high' && pct < 75)  return false
    if (scoreFilter === 'mid'  && (pct < 50 || pct >= 75)) return false
    if (scoreFilter === 'low'  && pct >= 50) return false
    if (sectorFilter !== 'all' && c.sector !== sectorFilter) return false
    if (decisionFilter !== 'all' && c.decision !== decisionFilter) return false
    return true
  })

  const counts = {
    high: candidates.filter(c => Math.round((c.score ?? 0) * 100) >= 75).length,
    mid:  candidates.filter(c => { const p = Math.round((c.score ?? 0) * 100); return p >= 50 && p < 75 }).length,
    low:  candidates.filter(c => Math.round((c.score ?? 0) * 100) < 50).length,
  }

  return (
    <div className="flex min-h-screen">
      {/* Sidebar Filters */}
      <aside className="w-72 flex-shrink-0 p-8 space-y-8">
        <div className="bg-surface-container-low p-6 rounded-3xl">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-on-surface">Filtres</h3>
            <button className="text-primary text-sm font-semibold" onClick={() => {
              setScoreFilter('all')
              setSectorFilter('all')
              setDecisionFilter('all')
            }}>Réinitialiser</button>
          </div>

          {/* Secteur */}
          {sectors.length > 0 && (
            <div className="mb-8">
              <label className="block text-sm font-bold text-on-surface mb-3">Secteur</label>
              <div className="space-y-2">
                {['all', ...sectors].map(s => (
                  <button key={s}
                    onClick={() => setSectorFilter(s)}
                    className={`w-full text-left px-4 py-2 rounded-xl text-sm font-medium transition-all
                      ${sectorFilter === s ? 'bg-white text-primary border border-primary/10' : 'text-on-surface-variant hover:bg-white'}`}>
                    {s === 'all' ? 'Tous les secteurs' : s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Décision */}
          <div className="mb-8">
            <label className="block text-sm font-bold text-on-surface mb-3">Décision</label>
            <div className="space-y-2">
              {[
                { id: 'all', label: 'Toutes' },
                { id: 'invite', label: 'Invités' },
                { id: 'reject', label: 'Rejetés' },
                { id: 'eliminated', label: 'Éliminés' },
              ].map(({ id, label }) => (
                <button key={id}
                  onClick={() => setDecisionFilter(id)}
                  className={`w-full text-left px-4 py-2 rounded-xl text-sm font-medium transition-all
                    ${decisionFilter === id ? 'bg-white text-primary border border-primary/10' : 'text-on-surface-variant hover:bg-white'}`}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Score IA */}
          <div>
            <label className="block text-sm font-bold text-on-surface mb-3">Score IA</label>
            <div className="space-y-3">
              {[
                { id: 'high', label: 'Top Match (≥75%)', count: counts.high },
                { id: 'mid',  label: 'Good Match (50-75%)', count: counts.mid },
                { id: 'low',  label: 'Low Match (<50%)', count: counts.low },
              ].map(({ id, label, count }) => (
                <button key={id}
                  onClick={() => setScoreFilter(scoreFilter === id ? 'all' : id)}
                  className={`w-full text-left px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center justify-between
                    ${scoreFilter === id ? 'bg-white text-tertiary border border-tertiary/10' : 'text-on-surface-variant hover:bg-white'}`}>
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
            <p className="text-sm opacity-90 leading-relaxed">
              {counts.high} candidat{counts.high > 1 ? 's' : ''} avec un score top match.
            </p>
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
              <span>Recrutement</span>
              <Icon name="chevron_right" size={14} />
              <span>Candidats</span>
            </nav>
            <h1 className="text-5xl font-black text-on-surface leading-tight tracking-tight">
              Tous les candidats
            </h1>
            <div className="flex items-center gap-4 mt-4">
              <span className="text-on-surface-variant text-sm font-medium flex items-center gap-1">
                <Icon name="group" size={16} /> {candidates.length} candidat{candidates.length > 1 ? 's' : ''} au total
              </span>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="bg-surface-container-lowest rounded-[2rem] overflow-hidden shadow-ambient">
          {loading ? (
            <div className="flex items-center justify-center py-24 text-on-surface-variant">
              <Icon name="hourglass_empty" size={32} />
              <span className="ml-3 text-sm font-medium">Chargement...</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-on-surface-variant gap-3">
              <Icon name="person_search" size={48} />
              <p className="text-sm font-medium">Aucun candidat trouvé</p>
              <button className="text-primary text-sm font-bold" onClick={() => { setScoreFilter('all'); setSectorFilter('all'); setDecisionFilter('all') }}>
                Réinitialiser les filtres
              </button>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low/50">
                  {['Candidat', 'Score IA', 'Date', 'Décision', ''].map(h => (
                    <th key={h} className="px-6 py-5 text-[11px] font-bold text-outline uppercase tracking-wider first:pl-8 last:pr-8 last:text-right">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((c, i) => {
                  const stage = decisionStage(c.decision)
                  const ini = initials(c.name)
                  const avatarColor = AVATAR_COLORS[i % AVATAR_COLORS.length]
                  const date = c.received_at ? new Date(c.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
                  return (
                    <tr key={c.candidate_id}
                      className="hover:bg-surface-container-low/30 transition-colors group cursor-pointer"
                      onClick={() => onNavigate('profile', c.candidate_id)}>
                      <td className="px-8 py-5">
                        <div className="flex items-center gap-4">
                          <div className={`w-12 h-12 rounded-2xl ${avatarColor} flex items-center justify-center font-bold flex-shrink-0 text-sm`}>
                            {ini}
                          </div>
                          <div>
                            <div className="font-bold text-on-surface">{c.name ?? 'Anonyme'}</div>
                            <div className="text-xs text-on-surface-variant">
                              {c.target_role ?? c.sector ?? '—'}{c.years_experience ? ` • ${c.years_experience} ans exp.` : ''}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-5"><ScoreRing score={c.score} /></td>
                      <td className="px-6 py-5 text-sm text-on-surface-variant font-medium">{date}</td>
                      <td className="px-6 py-5">
                        <span className={`px-4 py-1.5 ${stage.color} rounded-full text-xs font-bold flex items-center gap-2 w-fit`}>
                          <span className={`w-1.5 h-1.5 ${stage.dot} rounded-full`} />
                          {stage.label}
                        </span>
                      </td>
                      <td className="px-8 py-5 text-right">
                        <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant"><Icon name="visibility" size={18} /></button>
                          <button className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant"><Icon name="more_vert" size={18} /></button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}

          {/* Footer */}
          {!loading && filtered.length > 0 && (
            <div className="px-8 py-6 bg-surface-container-low/30 flex justify-between items-center">
              <span className="text-xs font-bold text-outline uppercase tracking-widest">
                {filtered.length} candidat{filtered.length > 1 ? 's' : ''} affichés sur {candidates.length}
              </span>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
