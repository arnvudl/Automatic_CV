import { useDraggable } from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'
import { Icon } from './Icon'

function initials(name = '') {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

function scoreStyle(score) {
  const pct = Math.round((score ?? 0) * 100)
  if (pct >= 75) return { label: `${pct}%`, cls: 'bg-success/10 text-success' }
  if (pct >= 50) return { label: `${pct}%`, cls: 'bg-warning/10 text-warning' }
  return              { label: `${pct}%`, cls: 'bg-destructive/10 text-destructive' }
}

const AVATAR_COLORS = [
  'bg-blue-100 text-blue-700',
  'bg-violet-100 text-violet-700',
  'bg-emerald-100 text-emerald-700',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

function avatarColor(id = '') {
  const n = id.charCodeAt(0) + id.charCodeAt(id.length - 1)
  return AVATAR_COLORS[n % AVATAR_COLORS.length]
}

export function KanbanCard({ candidate, onNavigate, onRemove, overlay = false }) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id:   candidate.candidate_id,
    data: { candidate },
  })

  const style = {
    transform: CSS.Translate.toString(transform),
    opacity:   isDragging ? 0.4 : 1,
    cursor:    isDragging ? 'grabbing' : 'grab',
  }

  const sc    = scoreStyle(candidate.score)
  const ini   = initials(candidate.name)
  const color = avatarColor(candidate.candidate_id)

  return (
    <div
      ref={overlay ? undefined : setNodeRef}
      style={overlay ? undefined : style}
      {...(overlay ? {} : { ...listeners, ...attributes })}
      className={`group relative bg-card border border-border rounded-lg p-3 shadow-card select-none
        ${overlay ? 'rotate-1 shadow-card-lg' : 'hover:shadow-card-md transition-shadow'}
        ${isDragging ? 'ring-2 ring-foreground/20' : ''}`}
    >
      {/* Bouton supprimer — visible au hover */}
      {!overlay && onRemove && (
        <button
          onPointerDown={e => e.stopPropagation()}
          onClick={() => onRemove(candidate.candidate_id)}
          className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity
            w-5 h-5 rounded-full bg-muted hover:bg-destructive/10 hover:text-destructive
            flex items-center justify-center text-muted-foreground"
          title="Retirer du pipeline"
        >
          <Icon name="close" size={12} />
        </button>
      )}

      <div className="flex items-start gap-2 mb-2">
        <div className={`w-7 h-7 rounded-md ${color} flex items-center justify-center text-[10px] font-bold flex-shrink-0`}>
          {ini}
        </div>
        <div className="flex-1 min-w-0 pr-4">
          <p className="text-xs font-semibold text-foreground truncate leading-tight">{candidate.name ?? 'Anonyme'}</p>
          <p className="text-[10px] text-muted-foreground truncate">{candidate.target_role ?? candidate.sector ?? '—'}</p>
        </div>
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0 ${sc.cls}`}>
          {sc.label}
        </span>
      </div>

      {candidate.sector && (
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-[9px] font-semibold text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
            {candidate.sector}
          </span>
          {candidate.years_experience != null && (
            <span className="text-[9px] font-semibold text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
              {candidate.years_experience} ans
            </span>
          )}
        </div>
      )}

      <button
        onPointerDown={e => e.stopPropagation()}
        onClick={() => onNavigate?.('profile', candidate.candidate_id)}
        className="mt-2 w-full flex items-center justify-center gap-1 text-[10px] font-semibold text-muted-foreground hover:text-foreground transition-colors py-0.5"
      >
        <Icon name="open_in_new" size={11} /> Voir le profil
      </button>
    </div>
  )
}
