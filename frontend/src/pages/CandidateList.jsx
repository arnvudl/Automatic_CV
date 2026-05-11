import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getCandidates, deleteCandidate } from '../lib/api'

const AVATAR_COLORS = [
  'bg-blue-100 text-blue-700',
  'bg-violet-100 text-violet-700',
  'bg-emerald-100 text-emerald-700',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

function initials(name) {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

function ScoreRing({ score }) {
  const pct = Math.round((score ?? 0) * 100)
  const color = pct >= 75 ? '#16a34a' : pct >= 50 ? '#ca8a04' : '#dc2626'
  const labelColor = pct >= 75 ? 'text-success' : pct >= 50 ? 'text-warning' : 'text-destructive'
  const badgeBg = pct >= 75 ? 'bg-success/10 text-success' : pct >= 50 ? 'bg-warning/10 text-warning' : 'bg-destructive/10 text-destructive'
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
  if (decision === 'invite')     return { label: 'Invité',    color: 'bg-success/10 text-success',          dot: 'bg-success' }
  if (decision === 'reject')     return { label: 'Rejeté',    color: 'bg-destructive/10 text-destructive',  dot: 'bg-destructive' }
  if (decision === 'eliminated') return { label: 'Éliminé',   color: 'bg-muted text-muted-foreground',      dot: 'bg-muted-foreground' }
  return                                { label: 'En attente', color: 'bg-muted text-muted-foreground',      dot: 'bg-muted-foreground' }
}

export default function CandidateList({ onNavigate }) {
  const [candidates, setCandidates]       = useState([])
  const [loading, setLoading]             = useState(true)
  const [scoreFilter, setScoreFilter]     = useState('all')
  const [sectorFilter, setSectorFilter]   = useState('all')
  const [decisionFilter, setDecisionFilter] = useState('all')
  const [deleting, setDeleting]           = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)

  useEffect(() => {
    getCandidates({ limit: 100 })
      .then(setCandidates)
      .catch(() => setCandidates([]))
      .finally(() => setLoading(false))
  }, [])

  const handleDelete = async (id) => {
    setDeleting(id)
    try {
      await deleteCandidate(id)
      setCandidates(prev => prev.filter(c => c.candidate_id !== id))
    } catch (_) {}
    finally { setDeleting(null); setConfirmDelete(null) }
  }

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

      {/* Confirm delete modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={() => setConfirmDelete(null)}>
          <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-destructive"><Icon name="warning" size={28} /></span>
              <h3 className="text-lg font-bold text-foreground">Supprimer le candidat ?</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-6">Cette action est irréversible. Le candidat sera supprimé de la base de données.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)}
                className="px-5 py-2 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors">
                Annuler
              </button>
              <button onClick={() => handleDelete(confirmDelete)} disabled={deleting === confirmDelete}
                className="px-5 py-2 rounded-lg bg-destructive text-white font-semibold text-sm hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2">
                {deleting === confirmDelete ? <Icon name="hourglass_empty" size={16} /> : <Icon name="delete" size={16} />}
                Supprimer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar Filters */}
      <aside className="w-72 flex-shrink-0 p-8 space-y-8">
        <div className="bg-card border border-border rounded-xl p-6 shadow-card">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-base font-bold text-foreground">Filtres</h3>
            <button className="text-foreground text-sm font-semibold hover:opacity-70 transition-opacity" onClick={() => {
              setScoreFilter('all')
              setSectorFilter('all')
              setDecisionFilter('all')
            }}>Réinitialiser</button>
          </div>

          {sectors.length > 0 && (
            <div className="mb-6">
              <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-3">Secteur</label>
              <div className="space-y-1">
                {['all', ...sectors].map(s => (
                  <button key={s} onClick={() => setSectorFilter(s)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all
                      ${sectorFilter === s
                        ? 'bg-foreground text-primary-foreground'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}>
                    {s === 'all' ? 'Tous les secteurs' : s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="mb-6">
            <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-3">Décision</label>
            <div className="space-y-1">
              {[
                { id: 'all',        label: 'Toutes' },
                { id: 'invite',     label: 'Invités' },
                { id: 'reject',     label: 'Rejetés' },
                { id: 'eliminated', label: 'Éliminés' },
              ].map(({ id, label }) => (
                <button key={id} onClick={() => setDecisionFilter(id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all
                    ${decisionFilter === id
                      ? 'bg-foreground text-primary-foreground'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-3">Score IA</label>
            <div className="space-y-2">
              {[
                { id: 'high', label: 'Top Match (≥75%)',   count: counts.high },
                { id: 'mid',  label: 'Good Match (50-75%)', count: counts.mid },
                { id: 'low',  label: 'Low Match (<50%)',    count: counts.low },
              ].map(({ id, label, count }) => (
                <button key={id} onClick={() => setScoreFilter(scoreFilter === id ? 'all' : id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-between
                    ${scoreFilter === id
                      ? 'bg-foreground text-primary-foreground'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}>
                  <span>{label}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${scoreFilter === id ? 'bg-white/20' : 'bg-muted'}`}>{count}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* AI Insights */}
        <div className="bg-foreground p-6 rounded-xl text-primary-foreground relative overflow-hidden">
          <div className="relative z-10">
            <Icon name="auto_awesome" fill size={32} />
            <h4 className="text-base font-bold mt-3 mb-2">Talent Pulse AI</h4>
            <p className="text-sm opacity-80 leading-relaxed">
              {counts.high} candidat{counts.high !== 1 ? 's' : ''} avec un score top match.
            </p>
          </div>
          <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-white/5 rounded-full blur-2xl" />
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 p-8 pl-0">
        <div className="mb-8 flex justify-between items-end">
          <div>
            <nav className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
              <span>Recrutement</span>
              <Icon name="chevron_right" size={14} />
              <span>Candidats</span>
            </nav>
            <h1 className="text-3xl font-bold text-foreground tracking-tight">Tous les candidats</h1>
            <div className="flex items-center gap-4 mt-2">
              <span className="text-muted-foreground text-sm flex items-center gap-1">
                <Icon name="group" size={16} /> {candidates.length} candidat{candidates.length !== 1 ? 's' : ''} au total
              </span>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-card">
          {loading ? (
            <div className="flex items-center justify-center py-24 text-muted-foreground gap-3">
              <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
              <span className="text-sm font-medium">Chargement...</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-muted-foreground gap-3">
              <Icon name="person_search" size={48} />
              <p className="text-sm font-medium">Aucun candidat trouvé</p>
              <button className="text-foreground text-sm font-bold hover:opacity-70 transition-opacity"
                onClick={() => { setScoreFilter('all'); setSectorFilter('all'); setDecisionFilter('all') }}>
                Réinitialiser les filtres
              </button>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-muted/50 border-b border-border">
                  {['Candidat', 'Score IA', 'Date', 'Décision', ''].map(h => (
                    <th key={h} className="px-6 py-4 text-[11px] font-bold text-muted-foreground uppercase tracking-wider first:pl-8 last:pr-8 last:text-right">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filtered.map((c, i) => {
                  const stage = decisionStage(c.decision)
                  const ini   = initials(c.name)
                  const avatarColor = AVATAR_COLORS[i % AVATAR_COLORS.length]
                  const date  = c.received_at
                    ? new Date(c.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
                    : '—'
                  return (
                    <tr key={c.candidate_id}
                      className="hover:bg-muted/30 transition-colors group cursor-pointer"
                      onClick={() => onNavigate('profile', c.candidate_id)}>
                      <td className="px-8 py-4">
                        <div className="flex items-center gap-4">
                          <div className={`w-10 h-10 rounded-lg ${avatarColor} flex items-center justify-center font-bold flex-shrink-0 text-sm`}>
                            {ini}
                          </div>
                          <div>
                            <div className="font-semibold text-foreground">{c.name ?? 'Anonyme'}</div>
                            <div className="text-xs text-muted-foreground">
                              {c.target_role ?? c.sector ?? '—'}{c.years_experience ? ` · ${c.years_experience} ans exp.` : ''}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4"><ScoreRing score={c.score} /></td>
                      <td className="px-6 py-4 text-sm text-muted-foreground">{date}</td>
                      <td className="px-6 py-4">
                        <span className={`px-3 py-1 ${stage.color} rounded-full text-xs font-semibold flex items-center gap-1.5 w-fit`}>
                          <span className={`w-1.5 h-1.5 ${stage.dot} rounded-full`} />
                          {stage.label}
                        </span>
                      </td>
                      <td className="px-8 py-4 text-right" onClick={e => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button onClick={() => setConfirmDelete(c.candidate_id)}
                            className="p-2 hover:bg-destructive/10 rounded-lg text-muted-foreground hover:text-destructive transition-colors">
                            <Icon name="delete" size={16} />
                          </button>
                          <button onClick={() => onNavigate('profile', c.candidate_id)}
                            className="p-2 hover:bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors">
                            <Icon name="visibility" size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}

          {!loading && filtered.length > 0 && (
            <div className="px-8 py-4 bg-muted/30 border-t border-border flex justify-between items-center">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                {filtered.length} candidat{filtered.length !== 1 ? 's' : ''} affichés sur {candidates.length}
              </span>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
