import { useState, useEffect, useCallback } from 'react'
import { Icon } from '../components/Icon'
import { getStats, getCandidates } from '../lib/api'
import { useRealtime } from '../hooks/useRealtime'

// ── Helpers ──────────────────────────────────────────────────────────
function scoreColor(score) {
  if (score >= 0.8) return 'text-tertiary'
  if (score >= 0.5) return 'text-yellow-600'
  return 'text-error'
}

function scoreBg(score) {
  if (score >= 0.8) return 'bg-tertiary-container/10 text-tertiary'
  if (score >= 0.5) return 'bg-yellow-100 text-yellow-700'
  return 'bg-error-container text-error'
}

function decisionLabel(d) {
  if (d === 'invite')     return '✓ Invité'
  if (d === 'reject')     return '✗ Rejeté'
  if (d === 'eliminated') return '⊘ Éliminé'
  return d ?? '—'
}

function initials(name = '') {
  return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

const AVATAR_COLORS = [
  'bg-blue-100 text-primary',
  'bg-secondary-fixed text-secondary',
  'bg-green-100 text-tertiary',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

// ── Component ─────────────────────────────────────────────────────────
export default function Dashboard({ onNavigate }) {
  const [stats, setStats]           = useState(null)
  const [candidates, setCandidates] = useState([])
  const [activity, setActivity]     = useState([])   // flux temps réel
  const [connected, setConnected]   = useState(false)
  const [loading, setLoading]       = useState(true)

  const refresh = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([
        getStats(),
        getCandidates({ limit: 5 }),
      ])
      setStats(s)
      setCandidates(c)
    } catch (err) {
      console.error('API error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // SSE — temps réel
  useRealtime({
    onConnected: () => setConnected(true),
    onCandidateScored: (data) => {
      // Ajoute en haut de la liste candidates
      setCandidates(prev => [data, ...prev].slice(0, 5))
      // Flux d'activité
      setActivity(prev => [{
        icon: 'person_add',
        color: 'text-tertiary',
        text: <><strong>{data.name || 'Candidat'}</strong> scoré — {decisionLabel(data.decision)}</>,
        meta: `Score ${Math.round((data.score ?? 0) * 100)}% • à l'instant`,
      }, ...prev].slice(0, 6))
      // Refresh stats
      getStats().then(setStats).catch(() => {})
    },
    onStatusUpdated: (data) => {
      setCandidates(prev =>
        prev.map(c => c.candidate_id === data.candidate_id ? { ...c, status: data.status } : c)
      )
    },
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh] gap-3 text-on-surface-variant">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <span className="text-sm font-medium">Chargement du dashboard…</span>
      </div>
    )
  }

  return (
    <div className="p-8 space-y-8">
      {/* Hero */}
      <section className="flex items-end justify-between">
        <div>
          <h1 className="text-6xl font-black leading-tight tracking-tight text-on-surface">Dashboard</h1>
          <p className="text-on-surface-variant text-lg mt-1">Bienvenue. Voici l'état de votre pipeline de recrutement.</p>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold
          ${connected ? 'bg-tertiary/10 text-tertiary' : 'bg-surface-container-high text-on-surface-variant'}`}>
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-tertiary animate-pulse' : 'bg-outline'}`} />
          {connected ? 'Temps réel connecté' : 'Connexion…'}
        </div>
      </section>

      {/* Stats Bento */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard icon="group" bg="bg-primary/10" iconColor="text-primary"
          value={stats?.total ?? 0} label="Candidats total" />
        <StatCard icon="check_circle" bg="bg-tertiary/10" iconColor="text-tertiary"
          value={stats?.invited ?? 0} label="Invités"
          trend={stats?.invite_rate ? `${stats.invite_rate}%` : null} trendColor="text-tertiary" />
        <StatCard icon="cancel" bg="bg-error/10" iconColor="text-error"
          value={stats?.rejected ?? 0} label="Rejetés" />
        <StatCard icon="event" bg="bg-secondary-container/20" iconColor="text-secondary"
          value={stats?.today ?? 0} label="Aujourd'hui" badge="Aujourd'hui" />
      </section>

      {/* Main Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Top Matches */}
        <section className="lg:col-span-8 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-[1.375rem] font-bold text-on-surface">Meilleurs profils IA</h2>
            <button onClick={() => onNavigate('candidates')} className="text-primary text-sm font-semibold hover:underline">
              Voir tous les candidats →
            </button>
          </div>

          {candidates.length === 0 ? (
            <EmptyState
              icon="upload_file"
              title="Aucun CV reçu"
              desc="Les candidats apparaîtront ici après traitement par n8n ou via /score."
            />
          ) : (
            <div className="space-y-4">
              {candidates.map((c, i) => (
                <CandidateCard key={c.candidate_id ?? i} candidate={c} index={i}
                  onClick={() => onNavigate('profile', c.candidate_id)} />
              ))}
            </div>
          )}
        </section>

        {/* Right column */}
        <section className="lg:col-span-4 space-y-6">
          {/* Activité temps réel */}
          <h2 className="text-[1.375rem] font-bold text-on-surface">Activité récente</h2>
          <div className="bg-surface-container-low rounded-3xl p-6">
            {activity.length === 0 ? (
              <p className="text-sm text-on-surface-variant text-center py-4">
                En attente d'événements…
              </p>
            ) : (
              <div className="space-y-6">
                {activity.map((a, i) => (
                  <div key={i} className="flex gap-4">
                    <div className="relative z-10 w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm flex-shrink-0">
                      <span className={a.color}><Icon name={a.icon} size={16} /></span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-on-surface">{a.text}</p>
                      <p className="text-xs text-on-surface-variant mt-1">{a.meta}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* AI Insight */}
          {stats?.borderline > 0 && (
            <div className="bg-gradient-to-br from-primary to-primary-container p-6 rounded-3xl text-white shadow-lg">
              <div className="flex items-center gap-3 mb-4">
                <Icon name="auto_awesome" fill size={22} />
                <span className="text-sm font-bold">Insight IA</span>
              </div>
              <p className="text-sm leading-relaxed opacity-90">
                <strong>{stats.borderline}</strong> candidat{stats.borderline > 1 ? 's' : ''} sont dans la zone grise (score ±8% du seuil). Une revue humaine est recommandée.
              </p>
              <div className="mt-4 pt-4 border-t border-white/20">
                <button onClick={() => onNavigate('candidates')} className="text-xs font-bold flex items-center gap-2 hover:opacity-80">
                  Consulter les borderlines <Icon name="arrow_forward" size={14} />
                </button>
              </div>
            </div>
          )}

          {/* Stats by sector */}
          {stats?.by_sector && Object.keys(stats.by_sector).length > 0 && (
            <div className="bg-surface-container-lowest rounded-3xl p-6 shadow-ambient">
              <h3 className="text-sm font-black text-on-surface uppercase tracking-widest mb-4">Par secteur</h3>
              <div className="space-y-3">
                {Object.entries(stats.by_sector).slice(0, 5).map(([sector, data]) => {
                  const rate = data.total > 0 ? Math.round(data.invited / data.total * 100) : 0
                  return (
                    <div key={sector}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-medium text-on-surface">{sector}</span>
                        <span className="text-xs font-bold text-on-surface-variant">{data.invited}/{data.total}</span>
                      </div>
                      <div className="w-full h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-primary to-primary-container rounded-full"
                          style={{ width: `${rate}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────
function StatCard({ icon, bg, iconColor, value, label, trend, trendColor, badge }) {
  return (
    <div className="bg-surface-container-lowest p-8 rounded-3xl shadow-ambient">
      <div className="flex justify-between items-start mb-4">
        <div className={`p-3 ${bg} rounded-2xl`}>
          <span className={iconColor}><Icon name={icon} size={24} /></span>
        </div>
        {trend && <span className={`${trendColor ?? 'text-tertiary'} text-sm font-bold flex items-center gap-1`}>
          <Icon name="trending_up" size={16} /> {trend}
        </span>}
        {badge && !trend && <span className="bg-surface-container-high px-3 py-1 rounded-full text-xs font-bold text-on-surface-variant">{badge}</span>}
      </div>
      <div className="text-5xl font-black text-on-surface">{value}</div>
      <div className="text-on-surface-variant font-medium mt-1">{label}</div>
    </div>
  )
}

function CandidateCard({ candidate: c, index, onClick }) {
  const pct  = Math.round((c.score ?? 0) * 100)
  const color = AVATAR_COLORS[index % AVATAR_COLORS.length]

  return (
    <div onClick={onClick}
      className="bg-surface-container-lowest p-6 rounded-2xl transition-all hover:shadow-ambient-lg hover:-translate-y-0.5 cursor-pointer">
      <div className="flex items-start gap-6">
        <div className={`w-16 h-16 rounded-2xl ${color} flex items-center justify-center text-xl font-bold flex-shrink-0`}>
          {initials(c.name)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex justify-between items-start gap-2">
            <div className="min-w-0">
              <h3 className="font-bold text-lg text-on-surface truncate">{c.name || 'Anonyme'}</h3>
              <p className="text-on-surface-variant text-sm font-medium truncate">
                {c.target_role || c.sector || '—'}{c.years_experience ? ` • ${c.years_experience} ans exp.` : ''}
              </p>
            </div>
            <div className={`${scoreBg(c.score ?? 0)} px-4 py-1 rounded-full flex items-center gap-1.5 flex-shrink-0`}>
              <Icon name="auto_awesome" fill size={16} />
              <span className="text-sm font-bold">{pct}%</span>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-3">
            <span className={`text-xs font-bold ${scoreColor(c.score ?? 0)}`}>{decisionLabel(c.decision)}</span>
            {c.sector && <span className="px-3 py-1 bg-surface-container text-on-surface-variant text-xs font-semibold rounded-lg">{c.sector}</span>}
            {c.status && c.status !== 'inbox' && (
              <span className="px-3 py-1 bg-primary/10 text-primary text-xs font-semibold rounded-lg capitalize">{c.status}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function EmptyState({ icon, title, desc }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4 text-on-surface-variant bg-surface-container-lowest rounded-3xl">
      <span className="text-on-surface-variant/30"><Icon name={icon} size={56} /></span>
      <h3 className="text-lg font-bold text-on-surface">{title}</h3>
      <p className="text-sm text-center max-w-xs">{desc}</p>
    </div>
  )
}
