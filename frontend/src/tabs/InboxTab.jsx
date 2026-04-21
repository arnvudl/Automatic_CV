import { useState } from 'react'
import { DndContext, closestCenter, PointerSensor, useSensor, useSensors, DragOverlay } from '@dnd-kit/core'
import { SortableContext, verticalListSortingStrategy, useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { GripVertical } from 'lucide-react'
import CandidateDetail from '../components/CandidateDetail'
import { getLocationCategory, getScoreColor } from '../utils/location'

const COLUMNS = [
  { id: 'inbox',     label: 'Inbox',     color: 'border-blue-300',    bg: 'bg-blue-50/60'    },
  { id: 'review',    label: 'En revue',  color: 'border-amber-300',   bg: 'bg-amber-50/60'   },
  { id: 'interview', label: 'Entretien', color: 'border-emerald-300', bg: 'bg-emerald-50/60' },
  { id: 'rejected',  label: 'Rejeté',    color: 'border-red-300',     bg: 'bg-red-50/60'     },
]

const COL_ICONS = {
  inbox:     <InboxSVG />, review: <SearchSVG />,
  interview: <CheckSVG />, rejected: <XSvg />,
}

async function updateStatus(id, status) {
  await fetch(`/candidates/${id}/status`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  })
}

export default function InboxTab({ candidates, onUpdate }) {
  const [active, setActive]     = useState(null)
  const [selected, setSelected] = useState(null)
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }))
  const enriched = candidates.map(c => ({ ...c, status: c.status || 'inbox' }))
  const grouped  = Object.fromEntries(COLUMNS.map(col => [
    col.id, enriched.filter(c => (c.status || 'inbox') === col.id)
  ]))

  async function handleDragEnd({ active: a, over }) {
    if (!over) return
    if (COLUMNS.some(c => c.id === over.id)) {
      await updateStatus(a.id, over.id); onUpdate()
    }
  }

  async function handleAction(id, status) {
    await updateStatus(id, status); onUpdate()
  }

  const activeCard = active ? enriched.find(c => c.candidate_id === active) : null

  return (
    <>
      <DndContext sensors={sensors} collisionDetection={closestCenter}
        onDragStart={({ active: a }) => setActive(a.id)}
        onDragEnd={(...args) => { handleDragEnd(...args); setActive(null) }}>
        <div className="grid grid-cols-4 gap-4 h-full">
          {COLUMNS.map(col => (
            <Column key={col.id} col={col} cards={grouped[col.id] || []} onCardClick={setSelected} />
          ))}
        </div>
        <DragOverlay>
          {activeCard && <MiniCard c={activeCard} />}
        </DragOverlay>
      </DndContext>
      {selected && (
        <CandidateDetail candidate={selected} onClose={() => setSelected(null)} onAction={handleAction} />
      )}
    </>
  )
}

function Column({ col, cards, onCardClick }) {
  const { setNodeRef } = useSortable({ id: col.id, data: { type: 'column' } })
  return (
    <div ref={setNodeRef}
      className={`${col.bg} glass rounded-2xl border-t-2 ${col.color} p-3 flex flex-col gap-2 min-h-[440px]`}>
      <div className="flex items-center justify-between px-1 pb-2 border-b border-slate-200/60">
        <span className="font-semibold text-sm text-slate-600 flex items-center gap-1.5">
          <span className="text-slate-400">{COL_ICONS[col.id]}</span>
          {col.label}
        </span>
        <span className="text-xs bg-white/80 border border-slate-200 rounded-full px-2 py-0.5 text-slate-500 font-medium">{cards.length}</span>
      </div>
      <SortableContext items={cards.map(c => c.candidate_id)} strategy={verticalListSortingStrategy}>
        {cards.map(c => <SortableCard key={c.candidate_id} c={c} onCardClick={onCardClick} />)}
      </SortableContext>
      {cards.length === 0 && (
        <div className="flex-1 flex items-center justify-center text-slate-300 text-sm border-2 border-dashed border-slate-200 rounded-xl m-1">
          Déposer ici
        </div>
      )}
    </div>
  )
}

function SortableCard({ c, onCardClick }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: c.candidate_id, data: { type: 'card', status: c.status },
  })
  return (
    <div ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.3 : 1 }}>
      <Card c={c} onCardClick={onCardClick} dragHandle={{ ...attributes, ...listeners }} />
    </div>
  )
}

function Card({ c, onCardClick, dragHandle }) {
  const loc   = getLocationCategory(c.phone)
  const score = parseFloat(c.score) || 0
  const pct   = Math.round(score * 100)
  return (
    <div className="bg-white/90 backdrop-blur rounded-xl shadow-sm border border-slate-200/80 p-3 cursor-pointer hover:shadow-md hover:border-blue-200 transition-all"
         onClick={() => onCardClick(c)}>
      <div className="flex items-start justify-between gap-1">
        <span className="font-semibold text-sm text-slate-800 truncate">{c.name || 'Inconnu'}</span>
        <span {...dragHandle} onClick={e => e.stopPropagation()}
              className="text-slate-300 hover:text-slate-500 cursor-grab flex-shrink-0 mt-0.5">
          <GripVertical size={14} />
        </span>
      </div>
      <p className="text-xs text-slate-400 truncate mt-0.5">{c.target_role || '—'}</p>
      <div className="flex items-center justify-between mt-2.5">
        <span className={`text-xs px-1.5 py-0.5 rounded-full ${loc.color}`}>{loc.label}</span>
        <div className="flex items-center gap-1.5">
          <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${score >= 0.6 ? 'bg-emerald-400' : score >= 0.4 ? 'bg-amber-400' : 'bg-red-400'}`}
                 style={{ width: `${pct}%` }} />
          </div>
          <span className={`text-xs font-bold ${score >= 0.6 ? 'text-emerald-600' : score >= 0.4 ? 'text-amber-600' : 'text-red-500'}`}>{pct}%</span>
        </div>
      </div>
    </div>
  )
}

function MiniCard({ c }) {
  return (
    <div className="bg-white rounded-xl shadow-lg border-2 border-blue-300 p-3 w-48 rotate-2">
      <p className="font-semibold text-sm text-slate-800 truncate">{c.name}</p>
      <p className="text-xs text-slate-400 truncate">{c.target_role}</p>
    </div>
  )
}

function InboxSVG() { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-6l-2 3H10l-2-3H2"/><path d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z"/></svg> }
function SearchSVG(){ return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> }
function CheckSVG() { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg> }
function XSvg()     { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> }
