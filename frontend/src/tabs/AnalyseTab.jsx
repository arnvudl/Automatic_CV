import { useState, useEffect, useCallback } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { pdf } from '@react-pdf/renderer'
import { ReportDocument } from '../components/ReportDocument'

const TOOLTIP = {
  contentStyle: {
    background: 'white', border: '1px solid #e2e8f0',
    borderRadius: 10, color: '#1e293b', fontSize: 12,
    boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
  }
}

function todayStr()    { return new Date().toISOString().slice(0, 10) }
function monthAgoStr() { const d = new Date(); d.setDate(d.getDate() - 30); return d.toISOString().slice(0, 10) }

export default function AnalyseTab({ candidates }) {
  const [start, setStart]               = useState(monthAgoStr())
  const [end, setEnd]                   = useState(todayStr())
  const [data, setData]                 = useState(null)
  const [loading, setLoading]           = useState(false)
  const [selected, setSelected]         = useState(null)
  const [explain, setExplain]           = useState(null)
  const [spotcheck, setSpotcheck]       = useState(null)
  const [spotLoading, setSpotLoading]   = useState(false)

  const load = useCallback(() => {
    setLoading(true)
    fetch(`/analyse/period?start=${start}&end=${end}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [start, end])

  useEffect(() => { load() }, [load])

  const loadExplain = (cid) => {
    setExplain(null); setSelected(cid)
    fetch(`/candidates/${cid}/explain`).then(r => r.json()).then(setExplain).catch(() => {})
  }

  const runSpotcheck = () => {
    setSpotLoading(true); setSpotcheck(null)
    fetch('/analyse/spotcheck?n=8').then(r => r.json()).then(d => { setSpotcheck(d); setSpotLoading(false) }).catch(() => setSpotLoading(false))
  }

  const exportPdf = async () => {
    if (!data) return
    const blob = await pdf(<ReportDocument data={data} start={start} end={end} />).toBlob()
    const url  = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `rapport_${start}_${end}.pdf`; a.click()
    URL.revokeObjectURL(url)
  }

  const shapBars = data ? Object.entries(data.shap_aggregate)
    .sort((a, b) => b[1] - a[1]).slice(0, 9)
    .map(([name, val]) => ({ name, val })) : []

  const compBars = data ? Object.entries(data.feature_comparison)
    .sort((a, b) => Math.abs(b[1].gap) - Math.abs(a[1].gap)).slice(0, 8)
    .map(([name, v]) => ({ name, invited: v.invited_avg, rejected: v.rejected_avg })) : []

  return (
    <div className="space-y-5">

      {/* Controls */}
      <div className="glass-card p-4 flex flex-wrap items-end gap-4">
        <div>
          <p className="text-xs text-slate-500 mb-1 font-medium">Début</p>
          <input type="date" value={start} onChange={e => setStart(e.target.value)}
            className="glass-input px-3 py-2 text-sm text-slate-700" />
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1 font-medium">Fin</p>
          <input type="date" value={end} onChange={e => setEnd(e.target.value)}
            className="glass-input px-3 py-2 text-sm text-slate-700" />
        </div>
        <button onClick={load} className="glass-btn-primary px-5 py-2 text-sm">
          Analyser
        </button>
        {data && (
          <button onClick={exportPdf} className="glass-btn-ghost px-5 py-2 text-sm flex items-center gap-2">
            <PdfIcon /> Exporter PDF
          </button>
        )}
        {data && (
          <div className="ml-auto flex gap-5 text-sm">
            <Stat label="Candidats"       value={data.total} />
            <Stat label="Invités"         value={data.invited}      color="text-emerald-600" />
            <Stat label="Rejetés"         value={data.rejected}     color="text-red-500" />
            <Stat label="Taux invitation" value={`${data.invite_rate}%`} color="text-blue-600" />
          </div>
        )}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-24 text-slate-400">Analyse en cours…</div>
      )}

      {/* Spotcheck panel */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-700">Audit pile rejetée — Spotcheck</h3>
            <p className="text-xs text-slate-400 mt-0.5">Échantillon aléatoire de candidats rejetés, triés par proximité au seuil. Détecte les faux négatifs potentiels.</p>
          </div>
          <button onClick={runSpotcheck} disabled={spotLoading}
            className="glass-btn-ghost px-4 py-2 text-sm flex items-center gap-2 flex-shrink-0">
            {spotLoading ? <span className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" /> : '🔍'}
            {spotLoading ? 'Analyse…' : 'Lancer le spotcheck'}
          </button>
        </div>

        {spotcheck && (
          <>
            <div className="flex gap-4 mb-3 text-xs text-slate-500">
              <span>Total rejetés : <strong>{spotcheck.n_rejected_total}</strong></span>
              <span>Échantillon : <strong>{spotcheck.n_sampled}</strong></span>
              {spotcheck.spotcheck.filter(s => s.suspicious).length > 0 && (
                <span className="text-amber-600 font-semibold">
                  ⚠ {spotcheck.spotcheck.filter(s => s.suspicious).length} profil(s) suspect(s)
                </span>
              )}
            </div>
            <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
              {spotcheck.spotcheck.map(s => (
                <div key={s.candidate_id}
                  className={`rounded-xl border p-3 transition-all
                    ${s.suspicious ? 'bg-amber-50 border-amber-200' : 'bg-white border-slate-100 hover:border-slate-200'}`}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-medium text-slate-800">{s.name}</span>
                        <span className="text-xs text-slate-400">{s.sector}</span>
                        {s.suspicious && (
                          <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-200 text-amber-800 font-semibold">Potentiel faux négatif</span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">{s.narrative}</p>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <span className="text-sm font-bold text-red-500">{Math.round(s.score * 100)}%</span>
                      <p className="text-xs text-slate-400">seuil {Math.round(s.threshold * 100)}%</p>
                      <p className="text-xs text-slate-300">Δ {Math.round(s.gap_to_threshold * 100)} pts</p>
                    </div>
                  </div>
                  {s.near_invite_features.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {s.near_invite_features.map((f, i) => (
                        <span key={i} className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-100 px-2 py-0.5 rounded-full">
                          {f.feature} : {f.pct_of_avg}% de la moy. invités
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {data && !loading && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-1">Variables les plus influentes</h3>
              <p className="text-xs text-slate-400 mb-3">Importance SHAP moyenne absolue sur la période</p>
              {shapBars.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={shapBars} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11, fill: '#475569' }} />
                    <Tooltip {...TOOLTIP} formatter={v => v.toFixed(4)} />
                    <Bar dataKey="val" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <NoShap />}
            </div>

            <div className="glass-card p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-1">Profil moyen : Invités vs Rejetés</h3>
              <p className="text-xs text-slate-400 mb-3">Valeurs moyennes par variable</p>
              {compBars.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={compBars} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11, fill: '#475569' }} />
                    <Tooltip {...TOOLTIP} />
                    <Bar dataKey="invited"  name="Invités"  fill="#86efac" radius={[0, 3, 3, 0]} />
                    <Bar dataKey="rejected" name="Rejetés"  fill="#fca5a5" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <NoShap />}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            {/* Candidate list */}
            <div className="col-span-2 glass-card p-5">
              <h3 className="text-sm font-semibold text-slate-700 mb-3">
                Candidats de la période ({data.candidates.length})
              </h3>
              <div className="space-y-1 max-h-[400px] overflow-y-auto pr-1">
                {data.candidates.length === 0 && (
                  <p className="text-slate-300 text-sm text-center py-8">Aucun candidat sur cette période</p>
                )}
                {data.candidates.map(c => (
                  <button key={c.candidate_id} onClick={() => loadExplain(c.candidate_id)}
                    className={`w-full text-left p-3 rounded-xl border transition-all
                      ${selected === c.candidate_id
                        ? 'bg-blue-50 border-blue-200'
                        : 'hover:bg-slate-50 border-transparent hover:border-slate-200'}`}>
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-slate-800">{c.name || '—'}</span>
                        <span className="text-slate-400 text-xs ml-2">{c.sector}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <ScorePill score={parseFloat(c.score)} />
                        <DecisionBadge decision={c.decision} />
                      </div>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 line-clamp-1">{c.narrative}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Drill-down */}
            <div className="glass-card p-5">
              {!selected ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-300 text-sm text-center gap-3 py-8">
                  <BrainSVG />
                  Sélectionner un candidat<br/>pour voir l'explication IA
                </div>
              ) : !explain ? (
                <div className="flex items-center justify-center h-full text-slate-400 text-sm">Chargement…</div>
              ) : (
                <ExplainPanel explain={explain} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function ExplainPanel({ explain }) {
  const shapEntries = Object.entries(explain.shap || {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 7)
  const maxAbs = Math.max(...shapEntries.map(([, v]) => Math.abs(v)), 0.001)

  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">Explication IA</p>
        <p className="text-xs text-slate-600 leading-relaxed">{explain.narrative}</p>
      </div>

      {shapEntries.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Impact par variable</p>
          <div className="space-y-1.5">
            {shapEntries.map(([name, val]) => (
              <div key={name}>
                <div className="flex justify-between text-xs mb-0.5">
                  <span className="text-slate-500">{name}</span>
                  <span className={val >= 0 ? 'text-emerald-600 font-medium' : 'text-red-500 font-medium'}>
                    {val >= 0 ? '+' : ''}{val.toFixed(3)}
                  </span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${val >= 0 ? 'bg-emerald-400' : 'bg-red-400'}`}
                       style={{ width: `${Math.abs(val) / maxAbs * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {explain.missing?.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Ce qui lui manque</p>
          <div className="space-y-1.5">
            {explain.missing.map((m, i) => (
              <div key={i} className="bg-red-50 border border-red-100 rounded-lg p-2.5">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-red-700">{m.feature}</span>
                  <span className="text-xs text-red-500 font-semibold">-{m.gap_pct}%</span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  {m.candidate_value} vs {m.invited_avg} (moy. invités)
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function NoShap() {
  return (
    <p className="text-slate-300 text-sm text-center py-8">
      Pas de données SHAP.<br/>
      <span className="text-xs">Calculées lors du scoring de nouveaux CVs.</span>
    </p>
  )
}

function Stat({ label, value, color = 'text-slate-700' }) {
  return (
    <div className="text-center">
      <div className={`text-base font-bold ${color}`}>{value ?? '—'}</div>
      <div className="text-slate-400 text-xs">{label}</div>
    </div>
  )
}

function ScorePill({ score }) {
  const pct = Math.round(score * 100)
  const cls = score >= 0.6 ? 'score-pill-high' : score >= 0.4 ? 'score-pill-mid' : 'score-pill-low'
  return <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${cls}`}>{pct}%</span>
}

function DecisionBadge({ decision }) {
  return decision === 'invite'
    ? <span className="text-xs px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Invité</span>
    : <span className="text-xs px-1.5 py-0.5 rounded-full bg-red-50 text-red-600 border border-red-200">Rejeté</span>
}

function BrainSVG() {
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M9.5 2A2.5 2.5 0 0112 4.5v15a2.5 2.5 0 01-4.96-.46 2.5 2.5 0 01-2.96-3.08 3 3 0 01-.34-5.58 2.5 2.5 0 011.32-4.84A2.5 2.5 0 019.5 2Z"/>
      <path d="M14.5 2A2.5 2.5 0 0112 4.5v15a2.5 2.5 0 004.96-.46 2.5 2.5 0 002.96-3.08 3 3 0 00.34-5.58 2.5 2.5 0 00-1.32-4.84A2.5 2.5 0 0014.5 2Z"/>
    </svg>
  )
}

function PdfIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="12" y1="18" x2="12" y2="12"/>
      <line x1="9" y1="15" x2="15" y2="15"/>
    </svg>
  )
}
