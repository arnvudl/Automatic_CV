import { useState } from 'react'
import { Icon } from './Icon'

export function InterviewModal({ candidate, onConfirm, onClose }) {
  const today = new Date().toISOString().split('T')[0]
  const [date,  setDate]  = useState(today)
  const [time,  setTime]  = useState('10:00')
  const [type,  setType]  = useState('Entretien technique')
  const [notes, setNotes] = useState('')

  const TYPES = [
    'Entretien RH',
    'Entretien technique',
    'Entretien final',
    'Présentation équipe',
    'Autre',
  ]

  const inputCls = "w-full bg-muted border border-border rounded-lg px-4 py-2.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 transition-all"

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg w-full max-w-md mx-4" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Icon name="event" size={20} className="text-foreground" />
            <h3 className="text-lg font-bold text-foreground">Planifier l'entretien</h3>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors">
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Candidate */}
        <div className="flex items-center gap-3 p-4 bg-muted rounded-lg mb-6">
          <div className="w-10 h-10 rounded-lg bg-card border border-border text-foreground flex items-center justify-center font-bold text-sm">
            {(candidate?.name ?? '??').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
          </div>
          <div>
            <p className="font-semibold text-foreground text-sm">{candidate?.name ?? 'Anonyme'}</p>
            <p className="text-xs text-muted-foreground">{candidate?.sector ?? '—'} · Score {Math.round((candidate?.score ?? 0) * 100)}%</p>
          </div>
        </div>

        {/* Form */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">Date</label>
              <input type="date" value={date} min={today}
                onChange={e => setDate(e.target.value)}
                className={inputCls} />
            </div>
            <div>
              <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">Heure</label>
              <input type="time" value={time}
                onChange={e => setTime(e.target.value)}
                className={inputCls} />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">Type d'entretien</label>
            <select value={type} onChange={e => setType(e.target.value)} className={inputCls}>
              {TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">Notes (optionnel)</label>
            <textarea value={notes} onChange={e => setNotes(e.target.value)}
              placeholder="Instructions, salle de réunion, lien visio..."
              rows={3}
              className={`${inputCls} resize-none`} />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-6 justify-end">
          <button onClick={onClose}
            className="px-5 py-2.5 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors">
            Annuler
          </button>
          <button onClick={() => onConfirm({ date, time, type, notes })}
            className="px-6 py-2.5 rounded-lg bg-foreground text-primary-foreground font-bold text-sm hover:opacity-90 transition-opacity flex items-center gap-2">
            <Icon name="event_available" size={16} />
            Confirmer l'entretien
          </button>
        </div>
      </div>
    </div>
  )
}
