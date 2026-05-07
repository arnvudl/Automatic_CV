import { useState } from 'react'
import { Icon } from './Icon'

export function InterviewModal({ candidate, onConfirm, onClose }) {
  const today = new Date().toISOString().split('T')[0]
  const [date, setDate]   = useState(today)
  const [time, setTime]   = useState('10:00')
  const [type, setType]   = useState('Entretien technique')
  const [notes, setNotes] = useState('')

  const TYPES = [
    'Entretien RH',
    'Entretien technique',
    'Entretien final',
    'Présentation équipe',
    'Autre',
  ]

  const handleConfirm = () => {
    onConfirm({ date, time, type, notes })
  }

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-surface-container-lowest rounded-3xl p-8 shadow-ambient-lg w-full max-w-md mx-4" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <span className="text-primary"><Icon name="event" size={24} /></span>
            <h3 className="text-xl font-extrabold text-on-surface">Planifier l'entretien</h3>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-surface-container-high rounded-xl text-on-surface-variant transition-colors">
            <Icon name="close" size={20} />
          </button>
        </div>

        {/* Candidat */}
        <div className="flex items-center gap-3 p-4 bg-surface-container-low rounded-2xl mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-bold text-sm">
            {(candidate?.name ?? '??').split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase()}
          </div>
          <div>
            <p className="font-bold text-on-surface text-sm">{candidate?.name ?? 'Anonyme'}</p>
            <p className="text-xs text-on-surface-variant">{candidate?.sector ?? '—'} · Score {Math.round((candidate?.score ?? 0) * 100)}%</p>
          </div>
        </div>

        {/* Formulaire */}
        <div className="space-y-4">
          {/* Date + Heure */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">Date</label>
              <input type="date" value={date} min={today}
                onChange={e => setDate(e.target.value)}
                className="w-full bg-surface-container-low border-none rounded-xl px-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all" />
            </div>
            <div>
              <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">Heure</label>
              <input type="time" value={time}
                onChange={e => setTime(e.target.value)}
                className="w-full bg-surface-container-low border-none rounded-xl px-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all" />
            </div>
          </div>

          {/* Type */}
          <div>
            <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">Type d'entretien</label>
            <select value={type} onChange={e => setType(e.target.value)}
              className="w-full bg-surface-container-low border-none rounded-xl px-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all">
              {TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">Notes (optionnel)</label>
            <textarea value={notes} onChange={e => setNotes(e.target.value)}
              placeholder="Instructions, salle de réunion, lien visio..."
              rows={3}
              className="w-full bg-surface-container-low border-none rounded-xl px-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all resize-none" />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-6 justify-end">
          <button onClick={onClose}
            className="px-5 py-2.5 rounded-xl bg-surface-container text-on-surface font-semibold text-sm hover:bg-surface-container-high transition-colors">
            Annuler
          </button>
          <button onClick={handleConfirm}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-sm hover:shadow-md active:scale-95 transition-all flex items-center gap-2">
            <Icon name="event_available" size={16} />
            Confirmer l'entretien
          </button>
        </div>
      </div>
    </div>
  )
}
