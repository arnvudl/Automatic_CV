import { useState, useEffect } from 'react'
import { Icon } from './Icon'
import { getScorecards, createScorecard, deleteScorecard } from '../lib/api'

// ── Étoiles cliquables ────────────────────────────────────────────────
function StarRating({ value, onChange, readonly = false }) {
  const [hovered, setHovered] = useState(0)
  const display = readonly ? value : (hovered || value)

  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map(n => (
        <button
          key={n}
          type="button"
          disabled={readonly}
          onClick={() => !readonly && onChange(n)}
          onMouseEnter={() => !readonly && setHovered(n)}
          onMouseLeave={() => !readonly && setHovered(0)}
          className={`transition-colors ${readonly ? 'cursor-default' : 'cursor-pointer'}`}
        >
          <Icon
            name="star"
            fill={n <= display}
            size={18}
            className={n <= display
              ? 'text-amber-400'
              : 'text-muted-foreground/30'}
          />
        </button>
      ))}
    </div>
  )
}

// ── Badge note globale ────────────────────────────────────────────────
function OverallBadge({ overall }) {
  if (overall == null) return null
  const color = overall >= 4 ? 'bg-emerald-100 text-emerald-700'
              : overall >= 3 ? 'bg-amber-100 text-amber-700'
              :                'bg-red-100 text-red-700'
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${color}`}>
      {overall.toFixed(1)} / 5
    </span>
  )
}

// ── Formulaire nouvelle évaluation ───────────────────────────────────
function ScorecardForm({ criteria, onSubmit, onCancel }) {
  const [evaluator, setEvaluator] = useState('')
  const [ratings, setRatings]     = useState({})
  const [notes, setNotes]         = useState('')
  const [saving, setSaving]       = useState(false)

  const allRated = criteria.every(c => ratings[c.key] >= 1)

  const handleSubmit = async () => {
    if (!allRated || saving) return
    setSaving(true)
    try {
      await onSubmit({ evaluator_name: evaluator || 'RH', ratings, notes })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-muted/40 border border-border rounded-xl p-5 space-y-4">
      {/* Nom évaluateur */}
      <div>
        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest block mb-1.5">
          Évaluateur
        </label>
        <input
          value={evaluator}
          onChange={e => setEvaluator(e.target.value)}
          placeholder="Votre nom…"
          className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20"
        />
      </div>

      {/* Critères */}
      <div className="space-y-3">
        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Critères</p>
        {criteria.map(c => (
          <div key={c.key} className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">{c.label}</span>
            <StarRating
              value={ratings[c.key] ?? 0}
              onChange={v => setRatings(prev => ({ ...prev, [c.key]: v }))}
            />
          </div>
        ))}
      </div>

      {/* Notes */}
      <div>
        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest block mb-1.5">
          Notes libres
        </label>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Impressions, points forts, réserves…"
          rows={3}
          className="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 resize-none"
        />
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-2">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm font-semibold text-muted-foreground hover:text-foreground transition-colors"
        >
          Annuler
        </button>
        <button
          onClick={handleSubmit}
          disabled={!allRated || saving}
          className="px-5 py-2 bg-foreground text-primary-foreground rounded-lg text-sm font-bold disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center gap-2"
        >
          {saving
            ? <><div className="w-3.5 h-3.5 border-2 border-primary-foreground/40 border-t-primary-foreground rounded-full animate-spin" /> Enregistrement…</>
            : <><Icon name="check" size={15} /> Enregistrer</>}
        </button>
      </div>
    </div>
  )
}

// ── Carte d'évaluation existante ──────────────────────────────────────
function ScorecardCard({ scorecard, criteria, onDelete }) {
  const [deleting, setDeleting] = useState(false)
  const date = scorecard.created_at
    ? new Date(scorecard.created_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
    : '—'

  const handleDelete = async () => {
    if (deleting) return
    setDeleting(true)
    try { await onDelete(scorecard.scorecard_id) }
    finally { setDeleting(false) }
  }

  return (
    <div className="bg-card border border-border rounded-xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center text-[10px] font-bold text-foreground">
            {(scorecard.evaluator_name ?? 'RH').slice(0, 2).toUpperCase()}
          </div>
          <div>
            <p className="text-sm font-bold text-foreground">{scorecard.evaluator_name ?? 'RH'}</p>
            <p className="text-[10px] text-muted-foreground">{date}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <OverallBadge overall={scorecard.overall} />
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="text-muted-foreground hover:text-destructive transition-colors disabled:opacity-40"
            title="Supprimer"
          >
            <Icon name="delete" size={15} />
          </button>
        </div>
      </div>

      {/* Critères */}
      <div className="space-y-2 mb-4">
        {criteria.map(c => (
          <div key={c.key} className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">{c.label}</span>
            <StarRating value={scorecard.ratings?.[c.key] ?? 0} readonly />
          </div>
        ))}
      </div>

      {/* Notes */}
      {scorecard.notes && (
        <p className="text-xs text-muted-foreground bg-muted rounded-lg px-3 py-2 leading-relaxed">
          {scorecard.notes}
        </p>
      )}
    </div>
  )
}

// ── Panneau principal ─────────────────────────────────────────────────
export function ScorecardPanel({ candidateId }) {
  const [criteria, setCriteria]       = useState([])
  const [scorecards, setScorecards]   = useState([])
  const [loading, setLoading]         = useState(true)
  const [showForm, setShowForm]       = useState(false)

  useEffect(() => {
    getScorecards(candidateId)
      .then(data => {
        setCriteria(data.criteria ?? [])
        setScorecards(data.scorecards ?? [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [candidateId])

  const handleSubmit = async (data) => {
    const created = await createScorecard(candidateId, data)
    setScorecards(prev => [created, ...prev])
    setShowForm(false)
  }

  const handleDelete = async (scorecardId) => {
    await deleteScorecard(scorecardId)
    setScorecards(prev => prev.filter(s => s.scorecard_id !== scorecardId))
  }

  // Moyenne globale de toutes les évaluations
  const avgOverall = scorecards.length
    ? (scorecards.reduce((s, c) => s + (c.overall ?? 0), 0) / scorecards.length).toFixed(1)
    : null

  return (
    <section className="bg-card border border-border rounded-xl p-8 shadow-card">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Icon name="assignment" fill size={22} className="text-foreground" />
          <div>
            <h3 className="text-lg font-bold text-foreground">Évaluation RH</h3>
            {avgOverall && (
              <p className="text-xs text-muted-foreground">
                Moyenne : <span className="font-bold text-foreground">{avgOverall} / 5</span>
                {' '}sur {scorecards.length} évaluation{scorecards.length > 1 ? 's' : ''}
              </p>
            )}
          </div>
        </div>
        {!showForm && (
          <button
            onClick={() => setShowForm(true)}
            className="flex items-center gap-2 px-4 py-2 bg-foreground text-primary-foreground rounded-lg text-sm font-bold hover:opacity-90 transition-opacity"
          >
            <Icon name="add" size={16} /> Nouvelle évaluation
          </button>
        )}
      </div>

      {/* Formulaire */}
      {showForm && (
        <div className="mb-5">
          <ScorecardForm
            criteria={criteria}
            onSubmit={handleSubmit}
            onCancel={() => setShowForm(false)}
          />
        </div>
      )}

      {/* Contenu */}
      {loading ? (
        <div className="flex items-center gap-2 text-muted-foreground py-4">
          <div className="w-4 h-4 border-2 border-border border-t-foreground rounded-full animate-spin" />
          <span className="text-sm">Chargement…</span>
        </div>
      ) : scorecards.length === 0 && !showForm ? (
        <div className="flex flex-col items-center justify-center py-10 text-muted-foreground gap-3">
          <Icon name="assignment" size={36} />
          <p className="text-sm">Aucune évaluation pour ce candidat</p>
          <button
            onClick={() => setShowForm(true)}
            className="text-sm font-semibold text-foreground underline underline-offset-2"
          >
            Créer la première évaluation
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {scorecards.map(sc => (
            <ScorecardCard
              key={sc.scorecard_id}
              scorecard={sc}
              criteria={criteria}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </section>
  )
}
