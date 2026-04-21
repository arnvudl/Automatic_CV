import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { AlertTriangle, CheckCircle, Clock, Users, TrendingUp, Activity } from 'lucide-react'

const PIPELINE_STEPS = [
  { id: 'received', label: 'Reçu',    icon: <InboxIcon /> },
  { id: 'parsed',   label: 'Parsé',   icon: <ScanIcon /> },
  { id: 'scored',   label: 'Scoré',   icon: <BrainIcon /> },
  { id: 'reviewed', label: 'En revue',icon: <EyeIcon /> },
  { id: 'decided',  label: 'Décidé',  icon: <CheckIcon /> },
]

const TOOLTIP = {
  contentStyle: {
    background: 'white', border: '1px solid #e2e8f0',
    borderRadius: 10, color: '#1e293b', fontSize: 12,
    boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
  }
}

export default function OverviewTab({ candidates }) {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetch('/stats').then(r => r.json()).then(setStats).catch(() => {})
  }, [candidates])

  if (!stats) return (
    <div className="flex items-center justify-center py-32 text-slate-400">Chargement…</div>
  )

  const scoreHist = buildScoreHistogram(candidates)
  const parsed    = candidates.length
  const scored    = candidates.filter(c => c.score).length
  const reviewed  = candidates.filter(c => ['review','interview','rejected'].includes(c.status)).length
  const decided   = candidates.filter(c => ['interview','rejected'].includes(c.status)).length
  const pipelineCounts = [stats.total, parsed, scored, reviewed, decided]

  return (
    <div className="space-y-5">

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <KpiCard icon={<Users size={15}/>}          label="Aujourd'hui"  value={stats.today}          accent="blue" />
        <KpiCard icon={<Activity size={15}/>}       label="Traités"      value={stats.total}          accent="indigo" />
        <KpiCard icon={<CheckCircle size={15}/>}    label="Invités"      value={stats.invited}        accent="emerald" sub={`${stats.invite_rate}%`} />
        <KpiCard icon={<TrendingUp size={15}/>}     label="Rejetés"      value={stats.rejected}       accent="red" />
        <KpiCard icon={<Clock size={15}/>}          label="En attente"   value={stats.pending_review} accent="amber" />
        <KpiCard icon={<AlertTriangle size={15}/>}  label="Borderline"   value={stats.borderline}     accent="orange" sub="à revoir" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Pipeline */}
        <div className="col-span-2 glass-card p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Statut du Pipeline</h3>
          <div className="flex items-center">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center gap-1.5 flex-1">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center
                    ${pipelineCounts[i] > 0
                      ? 'bg-blue-50 ring-2 ring-blue-200 text-blue-600'
                      : 'bg-slate-100 text-slate-400'}`}>
                    {step.icon}
                  </div>
                  <span className="text-xs text-slate-500 font-medium">{step.label}</span>
                  <span className="text-lg font-bold text-slate-800">{pipelineCounts[i]}</span>
                </div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <div className="w-8 h-px bg-slate-200 flex-shrink-0 mb-6" />
                )}
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-blue-50/60 rounded-xl flex items-center gap-2 text-xs text-slate-500 border border-blue-100">
            <CheckCircle size={12} className="text-emerald-500" />
            Parsing réussi : <strong className="text-slate-700">{scored > 0 ? Math.round(scored/Math.max(stats.total,1)*100) : 0}%</strong>
            &nbsp;·&nbsp; Score moyen : <strong className="text-slate-700">{Math.round((stats.avg_score||0)*100)}%</strong>
            &nbsp;·&nbsp; Modèle : <strong className="text-slate-700">v3 Fairness-Aware</strong>
          </div>
        </div>

        {/* Score histogram */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-3">Distribution des scores</h3>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={scoreHist} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="range" tick={{ fontSize: 8, fill: '#94a3b8' }} />
              <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} />
              <Tooltip {...TOOLTIP} formatter={(v, _, p) => [`${v} CVs`, p.payload.range]} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {scoreHist.map((e, i) => (
                  <Cell key={i} fill={e.mid >= 0.6 ? '#86efac' : e.mid >= 0.4 ? '#fde68a' : '#fca5a5'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alerts */}
      <div className="grid grid-cols-2 gap-4">
        <AlertBlock
          title="Cas borderline — révision recommandée"
          subtitle="Score à ±8% de la barre de sélection"
          accent="amber"
          items={stats.borderline_list}
          renderItem={c => (
            <div className="flex items-center justify-between py-2 border-b border-slate-50 last:border-0">
              <div>
                <span className="text-sm font-medium text-slate-700">{c.name || '—'}</span>
                <span className="text-slate-400 text-xs ml-2">{c.target_role}</span>
              </div>
              <div className="flex items-center gap-2">
                <ScorePill score={parseFloat(c.score)} />
                <span className="text-xs text-slate-400">barre {Math.round(parseFloat(c.threshold_used||0.5)*100)}%</span>
              </div>
            </div>
          )}
          emptyMsg="Aucun cas borderline"
        />
        <AlertBlock
          title="Profils intéressants rejetés"
          subtitle="Décision : rejeté · score > 30%"
          accent="blue"
          items={stats.interesting_rejected}
          renderItem={c => (
            <div className="flex items-center justify-between py-2 border-b border-slate-50 last:border-0">
              <div>
                <span className="text-sm font-medium text-slate-700">{c.name || '—'}</span>
                <span className="text-slate-400 text-xs ml-2">{c.sector}</span>
              </div>
              <ScorePill score={parseFloat(c.score)} />
            </div>
          )}
          emptyMsg="Aucun profil rejeté à revoir"
        />
      </div>
    </div>
  )
}

function buildScoreHistogram(candidates) {
  const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  const hist = bins.slice(0, -1).map((lo, i) => ({
    range: `${Math.round(lo*100)}-${Math.round(bins[i+1]*100)}`,
    mid: (lo + bins[i+1]) / 2,
    count: 0,
  }))
  candidates.forEach(c => {
    const s = parseFloat(c.score) || 0
    hist[Math.min(Math.floor(s * 10), 9)].count++
  })
  return hist
}

const ACCENT = {
  blue:    { border: 'border-blue-400',    icon: 'text-blue-500',    label: 'text-blue-600'    },
  indigo:  { border: 'border-indigo-400',  icon: 'text-indigo-500',  label: 'text-indigo-600'  },
  emerald: { border: 'border-emerald-400', icon: 'text-emerald-500', label: 'text-emerald-600' },
  red:     { border: 'border-red-400',     icon: 'text-red-500',     label: 'text-red-600'     },
  amber:   { border: 'border-amber-400',   icon: 'text-amber-500',   label: 'text-amber-600'   },
  orange:  { border: 'border-orange-400',  icon: 'text-orange-500',  label: 'text-orange-600'  },
}

function KpiCard({ icon, label, value, sub, accent = 'blue' }) {
  const a = ACCENT[accent]
  return (
    <div className={`glass-card p-4 border-t-2 ${a.border}`}>
      <div className={`flex items-center gap-1.5 mb-2 ${a.icon}`}>
        {icon}
        <span className="text-xs font-medium text-slate-500">{label}</span>
      </div>
      <p className="text-2xl font-bold text-slate-800">{value ?? '—'}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function AlertBlock({ title, subtitle, accent, items, renderItem, emptyMsg }) {
  return (
    <div className="glass-card p-5">
      <h3 className={`text-sm font-semibold ${accent === 'amber' ? 'text-amber-700' : 'text-blue-700'}`}>{title}</h3>
      <p className="text-xs text-slate-400 mb-3">{subtitle}</p>
      {items?.length > 0
        ? items.map((item, i) => <div key={i}>{renderItem(item)}</div>)
        : <p className="text-slate-300 text-sm text-center py-4">{emptyMsg}</p>}
    </div>
  )
}

function ScorePill({ score }) {
  const pct = Math.round(score * 100)
  const cls = score >= 0.6 ? 'score-pill-high' : score >= 0.4 ? 'score-pill-mid' : 'score-pill-low'
  return <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${cls}`}>{pct}%</span>
}

function InboxIcon()  { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-6l-2 3H10l-2-3H2"/><path d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z"/></svg> }
function ScanIcon()   { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 7V5a2 2 0 012-2h2"/><path d="M17 3h2a2 2 0 012 2v2"/><path d="M21 17v2a2 2 0 01-2 2h-2"/><path d="M7 21H5a2 2 0 01-2-2v-2"/><line x1="7" y1="12" x2="17" y2="12"/></svg> }
function BrainIcon()  { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.5 2A2.5 2.5 0 0112 4.5v15a2.5 2.5 0 01-4.96-.46 2.5 2.5 0 01-2.96-3.08 3 3 0 01-.34-5.58 2.5 2.5 0 011.32-4.84A2.5 2.5 0 019.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0112 4.5v15a2.5 2.5 0 004.96-.46 2.5 2.5 0 002.96-3.08 3 3 0 00.34-5.58 2.5 2.5 0 00-1.32-4.84A2.5 2.5 0 0014.5 2Z"/></svg> }
function EyeIcon()    { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> }
function CheckIcon()  { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg> }
