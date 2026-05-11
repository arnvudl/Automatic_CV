import { useState, useEffect, useRef } from 'react'
import { DndContext, DragOverlay, closestCorners, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { KanbanColumn } from './KanbanColumn'
import { KanbanCard } from './KanbanCard'
import { Icon } from './Icon'
import { moveCandidateStage, getCandidates } from '../lib/api'

// ── Modale de sélection de candidat ──────────────────────────────────
function CandidatePicker({ onSelect, onClose }) {
  const [query, setQuery]       = useState('')
  const [all, setAll]           = useState([])
  const [loading, setLoading]   = useState(true)
  const inputRef = useRef(null)

  useEffect(() => {
    getCandidates({ limit: 200 })
      .then(data => setAll(Array.isArray(data) ? data : data.candidates ?? []))
      .catch(() => setAll([]))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { inputRef.current?.focus() }, [])

  const filtered = query.trim()
    ? all.filter(c =>
        (c.name        ?? '').toLowerCase().includes(query.toLowerCase()) ||
        (c.target_role ?? '').toLowerCase().includes(query.toLowerCase()) ||
        (c.sector      ?? '').toLowerCase().includes(query.toLowerCase())
      )
    : all

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-card border border-border rounded-2xl shadow-xl w-[420px] max-h-[520px] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <p className="text-sm font-bold text-foreground">Ajouter un candidat</p>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Recherche */}
        <div className="px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2 bg-muted rounded-lg px-3 py-2">
            <Icon name="search" size={16} className="text-muted-foreground flex-shrink-0" />
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Rechercher un candidat…"
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none"
            />
            {query && (
              <button onClick={() => setQuery('')} className="text-muted-foreground hover:text-foreground">
                <Icon name="close" size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Liste */}
        <div className="flex-1 overflow-y-auto divide-y divide-border">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
              <div className="w-4 h-4 border-2 border-border border-t-foreground rounded-full animate-spin" />
              <span className="text-sm">Chargement…</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground gap-2">
              <Icon name="person_search" size={32} />
              <p className="text-sm">Aucun candidat trouvé</p>
            </div>
          ) : (
            filtered.slice(0, 50).map(c => {
              const score = c.score != null ? Math.round(c.score * 100) : null
              const scoreColor = score == null ? '' : score >= 70 ? 'text-emerald-600' : score >= 45 ? 'text-amber-500' : 'text-destructive'
              return (
                <button
                  key={c.candidate_id}
                  onClick={() => onSelect(c)}
                  className="w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/60 transition-colors text-left"
                >
                  {/* Avatar */}
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-foreground flex-shrink-0">
                    {(c.name ?? '?').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-foreground truncate">{c.name ?? 'Inconnu'}</p>
                    <p className="text-[11px] text-muted-foreground truncate">{c.target_role ?? c.sector ?? '—'}</p>
                  </div>
                  {score != null && (
                    <span className={`text-xs font-bold flex-shrink-0 ${scoreColor}`}>{score}%</span>
                  )}
                </button>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}

// ── Board principal ───────────────────────────────────────────────────
export function KanbanBoard({ stages: initialStages, onNavigate }) {
  const [stages, setStages]       = useState(initialStages)
  const [activeId, setActiveId]   = useState(null)
  const [pickerStageId, setPickerStageId] = useState(null)   // null = fermé

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } })
  )

  const activeCandidate = activeId
    ? stages.flatMap(s => s.candidates).find(c => c.candidate_id === activeId)
    : null

  const handleDragStart = ({ active }) => setActiveId(active.id)

  const handleDragEnd = ({ active, over }) => {
    setActiveId(null)
    if (!over || active.id === over.id) return

    const targetStageId = over.id
    const candidateId   = active.id

    const sourceStage = stages.find(s =>
      s.candidates.some(c => c.candidate_id === candidateId)
    )
    if (!sourceStage || sourceStage.stage_id === targetStageId) return

    setStages(prev => prev.map(s => {
      if (s.stage_id === sourceStage.stage_id)
        return { ...s, candidates: s.candidates.filter(c => c.candidate_id !== candidateId) }
      if (s.stage_id === targetStageId) {
        const moved = { ...sourceStage.candidates.find(c => c.candidate_id === candidateId), stage_id: targetStageId }
        return { ...s, candidates: [...s.candidates, moved] }
      }
      return s
    }))

    moveCandidateStage(candidateId, targetStageId).catch(() => setStages(initialStages))
  }

  // Retirer un candidat du pipeline (stage_id → null)
  const handleRemoveCandidate = (candidateId) => {
    setStages(prev => prev.map(s => ({
      ...s,
      candidates: s.candidates.filter(c => c.candidate_id !== candidateId),
    })))
    moveCandidateStage(candidateId, null).catch(() => setStages(initialStages))
  }

  // Ajout depuis la modale
  const handlePickCandidate = (candidate) => {
    const stageId = pickerStageId
    setPickerStageId(null)

    // Déjà dans ce stage ?
    const already = stages.some(s => s.candidates.some(c => c.candidate_id === candidate.candidate_id))
    if (already) {
      // Déplacer vers le nouveau stage
      setStages(prev => prev.map(s => ({
        ...s,
        candidates: s.stage_id === stageId
          ? [...s.candidates.filter(c => c.candidate_id !== candidate.candidate_id), { ...candidate, stage_id: stageId }]
          : s.candidates.filter(c => c.candidate_id !== candidate.candidate_id),
      })))
    } else {
      // Nouveau dans le board
      setStages(prev => prev.map(s =>
        s.stage_id === stageId
          ? { ...s, candidates: [...s.candidates, { ...candidate, stage_id: stageId }] }
          : s
      ))
    }

    moveCandidateStage(candidate.candidate_id, stageId).catch(() => setStages(initialStages))
  }

  return (
    <>
      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <div className="flex gap-4 overflow-x-auto pb-4">
          {stages.map(stage => (
            <KanbanColumn
              key={stage.stage_id}
              stage={stage}
              onNavigate={onNavigate}
              activeId={activeId}
              onAddCandidate={setPickerStageId}
              onRemoveCandidate={handleRemoveCandidate}
            />
          ))}
        </div>

        <DragOverlay>
          {activeCandidate ? <KanbanCard candidate={activeCandidate} overlay /> : null}
        </DragOverlay>
      </DndContext>

      {pickerStageId && (
        <CandidatePicker
          onSelect={handlePickCandidate}
          onClose={() => setPickerStageId(null)}
        />
      )}
    </>
  )
}
