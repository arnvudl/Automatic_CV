import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const GENDER_DATA  = [
  { name: 'Femmes', recall: 0.773, fill: '#f9a8d4' },
  { name: 'Hommes', recall: 0.786, fill: '#93c5fd' },
]
const AGE_DATA = [
  { name: 'Adulte (30+)', avant: 0.88, apres: 0.829 },
  { name: 'Jeune (<30)',  avant: 0.26, apres: 0.556 },
]
const COUNTRY_DATA = [
  { name: 'Allemagne',  recall: 0.917 }, { name: 'Portugal',   recall: 0.900 },
  { name: 'Inde',       recall: 0.833 }, { name: 'Pologne',    recall: 0.857 },
  { name: 'USA/Canada', recall: 0.800 }, { name: 'Irlande',    recall: 0.727 },
  { name: 'France',     recall: 0.700 }, { name: 'Italie',     recall: 0.714 },
  { name: 'Nigeria',    recall: 0.667 }, { name: 'Pays-Bas',   recall: 0.667 },
].sort((a, b) => b.recall - a.recall)

const SHAP_DATA = [
  { name: 'Niveau études',   shap: 0.529 }, { name: 'Prof. carrière',  shap: 0.288 },
  { name: 'Score potentiel', shap: 0.268 }, { name: 'Plurilingue',     shap: 0.203 },
  { name: 'Secteur IT',      shap: 0.151 }, { name: 'Field match',     shap: 0.116 },
  { name: 'Potentiel jr',    shap: 0.088 }, { name: 'Exp/âge',         shap: 0.057 },
  { name: 'Durée poste',     shap: 0.034 },
].sort((a, b) => a.shap - b.shap)

const PIE_DATA = [
  { name: 'Invités', value: 32, fill: '#86efac' },
  { name: 'Rejetés', value: 68, fill: '#fca5a5' },
]

const TOOLTIP = {
  contentStyle: {
    background: 'white', border: '1px solid #e2e8f0',
    borderRadius: 10, color: '#1e293b', fontSize: 12,
    boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
  }
}

export default function PerformanceTab() {
  const [stats, setStats] = useState(null)
  useEffect(() => {
    fetch('/stats').then(r => r.json()).then(setStats).catch(() => {})
  }, [])

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="AUC-ROC"       value="0.797"   sub="capacité de tri (max 1)"   accent="blue" />
        <KpiCard label="Écart genre"   value="1.3 pts" sub="Male vs Female recall"      accent="emerald" />
        <KpiCard label="Recall junior" value="55.6%"   sub="après correction barre"    accent="amber" />
        {stats
          ? <KpiCard label="Candidats live" value={stats.total} sub={`${stats.invite_rate}% invités`} accent="purple" />
          : <KpiCard label="Modèle"         value="v3"          sub="Fairness-Aware"                  accent="purple" />}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Équité Homme / Femme (Recall)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={GENDER_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748b' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <Tooltip {...TOOLTIP} formatter={v => `${Math.round(v * 100)}%`} />
              <Bar dataKey="recall" radius={[6, 6, 0, 0]}>
                {GENDER_DATA.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 text-center mt-1">Écart : 1.3 pts — v5 baseline : 13 pts</p>
        </Card>

        <Card title="Distribution Invités / Rejetés">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={PIE_DATA} cx="50%" cy="50%" outerRadius={75} dataKey="value"
                label={({ name, value }) => `${name} ${value}%`}>
                {PIE_DATA.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Pie>
              <Tooltip {...TOOLTIP} />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card title="Importance des variables (SHAP — modèle agrégé)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={SHAP_DATA} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis type="category" dataKey="name" width={105} tick={{ fontSize: 11, fill: '#475569' }} />
              <Tooltip {...TOOLTIP} formatter={v => v.toFixed(3)} />
              <Bar dataKey="shap" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Équité par Âge (avant / après correction barre)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={AGE_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <Tooltip {...TOOLTIP} formatter={v => `${Math.round(v * 100)}%`} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#64748b' }} />
              <Bar dataKey="avant" name="Avant" fill="#fca5a5" radius={[4, 4, 0, 0]} />
              <Bar dataKey="apres" name="Après" fill="#86efac" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 text-center mt-1">Junior : recall 0.26 → 0.56</p>
        </Card>
      </div>

      <Card title="Recall par Pays">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={COUNTRY_DATA}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
            <Tooltip {...TOOLTIP} formatter={v => `${Math.round(v * 100)}%`} />
            <Bar dataKey="recall" radius={[4, 4, 0, 0]}>
              {COUNTRY_DATA.map((e, i) => (
                <Cell key={i} fill={e.recall >= 0.8 ? '#86efac' : e.recall >= 0.65 ? '#fde68a' : '#fca5a5'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

const ACCENT = {
  blue:    'border-blue-400 text-blue-600',
  emerald: 'border-emerald-400 text-emerald-600',
  amber:   'border-amber-400 text-amber-600',
  purple:  'border-purple-400 text-purple-600',
}

function KpiCard({ label, value, sub, accent = 'blue' }) {
  return (
    <div className={`glass-card p-4 border-t-2 ${ACCENT[accent]}`}>
      <p className={`text-xs font-medium mb-1 ${ACCENT[accent]}`}>{label}</p>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      <p className="text-xs text-slate-400 mt-0.5">{sub}</p>
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-slate-700 mb-3">{title}</h3>
      {children}
    </div>
  )
}
