import { useDroppable } from '@dnd-kit/core'
import { KanbanCard } from './KanbanCard'
import { Icon } from './Icon'

export function KanbanColumn({ stage, onNavigate, activeId, onAddCandidate, onRemoveCandidate }) {
  const { setNodeRef, isOver } = useDroppable({ id: stage.stage_id })

  const isEmpty = stage.candidates.length === 0

  return (
    <div className="flex flex-col w-64 flex-shrink-0">
      {/* Column header */}
      <div className="flex items-center gap-2 mb-3 px-1">
        <span
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: stage.color }}
        />
        <h3 className="text-xs font-bold text-foreground truncate flex-1">{stage.name}</h3>
        <span className="text-[10px] font-bold text-muted-foreground bg-muted px-1.5 py-0.5 rounded-full min-w-[20px] text-center">
          {stage.candidates.length}
        </span>
      </div>

      {/* Drop zone */}
      <div
        ref={setNodeRef}
        className={`flex-1 min-h-[200px] rounded-xl p-2 space-y-2 transition-colors
          ${isOver
            ? 'bg-foreground/5 ring-2 ring-foreground/20'
            : 'bg-muted/40'}`}
      >
        {isEmpty && !isOver && (
          <div className="flex flex-col items-center justify-center h-24 text-muted-foreground/40">
            <span className="text-xl">↓</span>
            <p className="text-[10px] font-medium mt-1">Déposez ici</p>
          </div>
        )}
        {stage.candidates
          .filter(c => c.candidate_id !== activeId)
          .map(candidate => (
            <KanbanCard
              key={candidate.candidate_id}
              candidate={candidate}
              onNavigate={onNavigate}
              onRemove={onRemoveCandidate}
            />
          ))}
      </div>

      {/* Bouton ajouter un candidat */}
      <button
        onPointerDown={e => e.stopPropagation()}
        onClick={() => onAddCandidate(stage.stage_id)}
        className="mt-2 w-full flex items-center justify-center gap-1.5 py-2 rounded-lg border border-dashed border-border text-[11px] font-semibold text-muted-foreground hover:text-foreground hover:border-foreground/30 hover:bg-muted/60 transition-all"
      >
        <Icon name="add" size={14} /> Ajouter un candidat
      </button>
    </div>
  )
}
