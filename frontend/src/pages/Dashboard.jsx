import { useState, useEffect, useCallback } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { Icon } from '../components/Icon'
import { getStats, getCandidates } from '../lib/api'
import { useRealtime } from '../hooks/useRealtime'

const PERIODS = ['24H', '7J', '1M', '12M']

function decisionLabel(d) {
  if (d === 'invite')     return 'Invité'
  if (d === 'reject')     return 'Rejeté'
  if (d === 'eliminated') return 'Éliminé'
  return 'En attente'
}

function decisionStyle(d) {
  if (d === 'invite')  return 'text-success bg-success/10'
  if (d === 'reject')  return 'text-destructive bg-destructive/10'
  return 'text-muted-foreground bg-muted'
}

function initials(name = '') {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

const TOOLTIP_STYLE = {
  contentStyle: {
    background: 'hsl(0, 0%, 100%)',
    border: '1px solid hsl(220, 13%, 91%)',
    borderRadius: 8,
    fontSize: 12,
    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
  },
  cursor: { fill: 'hsl(220, 14%, 96%)' },
}

export default function Dashboard({ onNavigate }) {
  const [period, setPeriod]           = useState('1M')
  const [stats, setStats]             = useState(null)
  const [candidates, setCandidates]   = useState([])
  const [connected, setConnected]     = useState(false)
  const [loading, setLoading]         = useState(true)
  const [pipelineTab, setPipelineTab] = useState('pending')

  const refresh = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([getStats(), getCandidates({ limit: 5 })])
      setStats(s)
      setCandidates(c)
    } catch (err) {
      console.error('API error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useRealtime({
    onConnected: () => setConnected(true),
    onCandidateScored: (data) => {
      setCandidates(prev => [data, ...prev].slice(0, 5))
      getStats().then(setStats).catch(() => {})
    },
    onStatusUpdated: (data) => {
      setCandidates(prev =>
        prev.map(c => c.candidate_id === data.candidate_id ? { ...c, status: data.status } : c)
      )
    },
  })

  const inviteRate = stats ? Math.round(stats.invite_rate ?? 0) : 0
  const donutData = [
    { name: 'Invités', value: inviteRate },
    { name: 'Reste',   value: Math.max(0, 100 - inviteRate) },
  ]

  const sectorData = stats?.by_sector
    ? Object.entries(stats.by_sector).slice(0, 6).map(([sector, data]) => ({
        name:     sector.length > 10 ? sector.slice(0, 10) + '…' : sector,
        invited:  data.invited,
        rejected: (data.total ?? 0) - (data.invited ?? 0),
      }))
    : []

  return (
    <div className="p-8 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-1">Vue d'ensemble</p>
          <h1 className="text-2xl font-bold text-foreground">Tableau de bord</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold
            ${connected ? 'bg-success/10 text-success' : 'bg-muted text-muted-foreground'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-success animate-pulse' : 'bg-muted-foreground'}`} />
            {connected ? 'Temps réel' : 'Connexion…'}
          </div>
          <div className="flex bg-muted rounded-lg p-0.5">
            {PERIODS.map(p => (
              <button key={p} onClick={() => setPeriod(p)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all
                  ${period === p ? 'bg-card text-foreground shadow-card' : 'text-muted-foreground hover:text-foreground'}`}>
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Row 1 — 3 stat cards */}
      <div className="grid grid-cols-3 gap-6">

        {/* Total candidats */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-card">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-medium text-muted-foreground">Total candidats</p>
            <div className="w-8 h-8 bg-muted rounded-lg flex items-center justify-center text-muted-foreground">
              <Icon name="group" size={16} />
            </div>
          </div>
          <div className="flex items-end gap-3">
            <span className="text-4xl font-bold text-foreground">
              {loading ? '—' : (stats?.total ?? 0)}
            </span>
            {(stats?.today ?? 0) > 0 && (
              <span className="flex items-center gap-1 text-sm font-semibold text-success mb-1">
                <Icon name="trending_up" size={14} /> +{stats.today} aujourd'hui
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {stats?.invited ?? 0} invités · {stats?.rejected ?? 0} rejetés
          </p>
        </div>

        {/* Pipeline — dark card */}
        <div className="bg-foreground rounded-xl p-6 shadow-card">
          <div className="flex gap-1 mb-4">
            {[
              { id: 'pending', label: 'En attente' },
              { id: 'invited', label: 'Invités' },
            ].map(tab => (
              <button key={tab.id} onClick={() => setPipelineTab(tab.id)}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all
                  ${pipelineTab === tab.id ? 'bg-white/15 text-white' : 'text-white/50 hover:text-white/80'}`}>
                {tab.label}
              </button>
            ))}
          </div>
          <div className="text-4xl font-bold text-white mb-1">
            {loading ? '—' : (pipelineTab === 'invited' ? (stats?.invited ?? 0) : (stats?.pending_review ?? 0))}
          </div>
          <p className="text-white/60 text-xs">
            {pipelineTab === 'invited'
              ? 'candidats invités en entretien'
              : 'candidats en attente de revue'}
          </p>
          <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between text-xs">
            <span className="text-white/50">Zone grise</span>
            <span className="text-warning font-semibold">{stats?.borderline ?? 0} à revoir</span>
          </div>
        </div>

        {/* Taux d'invitation — donut */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-card">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-medium text-muted-foreground">Taux d'invitation</p>
            <Icon name="auto_awesome" fill size={16} className="text-muted-foreground" />
          </div>
          <div className="flex items-center gap-4">
            <div className="relative w-20 h-20 flex-shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donutData} cx="50%" cy="50%"
                    innerRadius={26} outerRadius={36}
                    startAngle={90} endAngle={-270}
                    dataKey="value" strokeWidth={0}
                  >
                    <Cell fill="hsl(142, 71%, 45%)" />
                    <Cell fill="hsl(220, 14%, 91%)" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-foreground">
                {loading ? '—' : `${inviteRate}%`}
              </span>
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">{inviteRate}%</p>
              <p className="text-xs text-muted-foreground">sur {stats?.total ?? 0} CVs</p>
              <p className={`text-xs font-semibold mt-1 ${inviteRate >= 30 ? 'text-success' : inviteRate >= 15 ? 'text-warning' : 'text-destructive'}`}>
                {inviteRate >= 30 ? '● Excellent' : inviteRate >= 15 ? '● Correct' : '● À améliorer'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2 — Chart + Candidates */}
      <div className="grid grid-cols-12 gap-6">

        {/* Bar chart by sector */}
        <div className="col-span-8 bg-card border border-border rounded-xl p-6 shadow-card">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-sm font-semibold text-foreground">Pipeline par secteur</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Candidats invités vs rejetés</p>
            </div>
            <button onClick={() => onNavigate('candidates')}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors">
              Voir tous →
            </button>
          </div>

          {sectorData.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-44 text-muted-foreground gap-2">
              <Icon name="bar_chart" size={32} />
              <p className="text-sm">Aucune donnée secteur disponible</p>
              <p className="text-xs opacity-60">Les candidats scorés apparaîtront ici</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sectorData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }} barGap={3}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'hsl(220, 9%, 46%)' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: 'hsl(220, 9%, 46%)' }} axisLine={false} tickLine={false} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="invited"  name="Invités"  fill="hsl(142, 71%, 45%)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rejected" name="Rejetés"  fill="hsl(220, 14%, 86%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Recent candidates — transaction style */}
        <div className="col-span-4 bg-card border border-border rounded-xl p-6 shadow-card flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-foreground">Candidats récents</h2>
            <button onClick={() => onNavigate('candidates')}
              className="text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors">
              Voir tous →
            </button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-10">
              <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
            </div>
          ) : candidates.length === 0 ? (
            <div className="flex flex-col items-center justify-center flex-1 gap-2 text-muted-foreground py-8">
              <Icon name="upload_file" size={28} />
              <p className="text-xs text-center">Aucun CV reçu.<br />Les candidats apparaîtront ici.</p>
            </div>
          ) : (
            <div className="space-y-1 flex-1">
              {candidates.map((c) => {
                const pct = Math.round((c.score ?? 0) * 100)
                return (
                  <button key={c.candidate_id}
                    onClick={() => onNavigate('profile', c.candidate_id)}
                    className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-muted transition-colors text-left">
                    <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center text-xs font-bold text-foreground flex-shrink-0">
                      {initials(c.name)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-foreground truncate">{c.name ?? 'Anonyme'}</p>
                      <p className="text-[10px] text-muted-foreground truncate">{c.sector ?? '—'}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span className="text-xs font-bold text-foreground">{pct}%</span>
                      <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${decisionStyle(c.decision)}`}>
                        {decisionLabel(c.decision)}
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          )}

          {(stats?.borderline ?? 0) > 0 && (
            <div className="mt-4 p-3 bg-warning/10 rounded-lg border border-warning/20">
              <div className="flex items-center gap-2 mb-1">
                <Icon name="auto_awesome" fill size={13} />
                <span className="text-xs font-semibold text-foreground">Insight IA</span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                {stats.borderline} candidat{stats.borderline > 1 ? 's' : ''} en zone grise — revue recommandée.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
