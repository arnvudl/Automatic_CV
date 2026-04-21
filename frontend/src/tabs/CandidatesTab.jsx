import { useState } from 'react'
import { Search, ChevronUp, ChevronDown } from 'lucide-react'
import CandidateDetail from '../components/CandidateDetail'
import { getLocationCategory, getCountry } from '../utils/location'

const EDU = { 1: 'Bac', 2: 'Bachelor', 3: 'Master', 4: 'PhD' }
const DECISIONS = [
  { value: '', label: 'Toutes décisions' },
  { value: 'invite', label: 'Invité' },
  { value: 'reject', label: 'Rejeté' },
]
const SECTORS = ['', 'IT', 'Finance', 'Industry', 'Public', 'Other']

export default function CandidatesTab({ candidates }) {
  const [search, setSearch]     = useState('')
  const [decision, setDecision] = useState('')
  const [sector, setSector]     = useState('')
  const [sort, setSort]         = useState({ key: 'score', dir: -1 })
  const [selected, setSelected] = useState(null)

  const filtered = candidates
    .filter(c => !decision || c.decision === decision)
    .filter(c => !sector   || c.sector === sector)
    .filter(c => !search   || [c.name, c.target_role, c.email, getCountry(c.phone)]
      .some(v => (v || '').toLowerCase().includes(search.toLowerCase())))
    .sort((a, b) => {
      const va = parseFloat(a[sort.key]) || 0
      const vb = parseFloat(b[sort.key]) || 0
      return sort.dir * (vb - va)
    })

  const toggleSort = key =>
    setSort(s => s.key === key ? { key, dir: -s.dir } : { key, dir: -1 })

  const SortIcon = ({ col }) => sort.key === col
    ? (sort.dir === -1 ? <ChevronDown size={11} /> : <ChevronUp size={11} />)
    : null

  return (
    <>
      <div className="flex gap-3 mb-4 flex-wrap items-center">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Rechercher nom, rôle, pays…"
            className="glass-input w-full pl-8 pr-3 py-2 text-sm" />
        </div>
        <LightSelect value={decision} onChange={setDecision} options={DECISIONS} />
        <LightSelect value={sector}   onChange={setSector}
          options={SECTORS.map(s => ({ value: s, label: s || 'Tous secteurs' }))} />
        <span className="text-xs text-slate-400">{filtered.length} candidat(s)</span>
      </div>

      <div className="glass rounded-2xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200/80 bg-slate-50/60">
              <Th>Nom</Th>
              <Th>Rôle</Th>
              <Th>Localisation</Th>
              <Th sortable onClick={() => toggleSort('score')}>Score <SortIcon col="score" /></Th>
              <Th sortable onClick={() => toggleSort('years_experience')}>Exp. <SortIcon col="years_experience" /></Th>
              <Th>Formation</Th>
              <Th>Décision</Th>
              <Th>Reçu</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filtered.map(c => {
              const loc   = getLocationCategory(c.phone)
              const score = parseFloat(c.score) || 0
              return (
                <tr key={c.candidate_id} onClick={() => setSelected(c)}
                  className="hover:bg-blue-50/50 cursor-pointer transition-colors">
                  <td className="px-4 py-3 font-medium text-slate-800">{c.name || '—'}</td>
                  <td className="px-4 py-3 text-slate-500">{c.target_role || '—'}</td>
                  <td className="px-4 py-3">
                    <span className={`text-xs px-2 py-0.5 rounded-full ${loc.color}`}>{loc.label}</span>
                  </td>
                  <td className="px-4 py-3">
                    <ScorePill score={score} />
                  </td>
                  <td className="px-4 py-3 text-slate-500">{c.years_experience ? `${c.years_experience}y` : '—'}</td>
                  <td className="px-4 py-3 text-slate-500">{EDU[parseInt(c.education_level)] || '—'}</td>
                  <td className="px-4 py-3">
                    <DecisionBadge decision={c.decision} />
                  </td>
                  <td className="px-4 py-3 text-slate-400 text-xs">
                    {c.received_at ? new Date(c.received_at).toLocaleDateString('fr-FR') : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="text-center py-12 text-slate-400">Aucun candidat trouvé</div>
        )}
      </div>

      {selected && (
        <CandidateDetail candidate={selected} onClose={() => setSelected(null)}
          onAction={() => setSelected(null)} />
      )}
    </>
  )
}

function Th({ children, sortable, onClick }) {
  return (
    <th className={`px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wide
      ${sortable ? 'cursor-pointer hover:text-slate-700 select-none' : ''}`} onClick={onClick}>
      <span className="flex items-center gap-1">{children}</span>
    </th>
  )
}

function LightSelect({ value, onChange, options }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)}
      className="glass-input text-sm px-3 py-2 cursor-pointer text-slate-700">
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

function ScorePill({ score }) {
  const pct = Math.round(score * 100)
  const cls = score >= 0.6 ? 'score-pill-high' : score >= 0.4 ? 'score-pill-mid' : 'score-pill-low'
  return <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${cls}`}>{pct}%</span>
}

function DecisionBadge({ decision }) {
  return decision === 'invite'
    ? <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Invité</span>
    : <span className="text-xs px-2 py-0.5 rounded-full bg-red-50 text-red-600 border border-red-200">Rejeté</span>
}
