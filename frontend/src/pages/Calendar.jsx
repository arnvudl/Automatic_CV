import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getInterviews, deleteInterview } from '../lib/api'
import { Toast } from '../components/Toast'

const DAYS_OF_WEEK = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
const MONTHS_FR    = ['Janvier','Février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre']

const TYPE_COLORS = {
  'Entretien RH':         'bg-primary/10 text-primary',
  'Entretien technique':  'bg-tertiary/10 text-tertiary',
  'Entretien final':      'bg-purple-100 text-purple-700',
  'Présentation équipe':  'bg-amber-100 text-amber-700',
  'Autre':                'bg-surface-container text-on-surface-variant',
}
const typeColor = (t) => TYPE_COLORS[t] ?? TYPE_COLORS['Autre']

function initials(name) {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

export default function Calendar({ onNavigate }) {
  const now   = new Date()
  const [year,  setYear]  = useState(now.getFullYear())
  const [month, setMonth] = useState(now.getMonth())
  const [interviews, setInterviews] = useState([])
  const [loading, setLoading] = useState(true)
  const [toast, setToast] = useState(null)
  const [selectedDay, setSelectedDay] = useState(null)

  const monthStr = `${year}-${String(month + 1).padStart(2, '0')}`

  useEffect(() => {
    setLoading(true)
    getInterviews({ month: monthStr })
      .then(setInterviews)
      .catch(() => setInterviews([]))
      .finally(() => setLoading(false))
  }, [monthStr])

  const today       = now.getDate()
  const isThisMonth = year === now.getFullYear() && month === now.getMonth()

  const firstDay      = new Date(year, month, 1).getDay()
  const leadingBlanks = (firstDay + 6) % 7
  const daysInMonth   = new Date(year, month + 1, 0).getDate()

  const cells = [
    ...Array(leadingBlanks).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ]
  while (cells.length % 7 !== 0) cells.push(null)

  // Group by day
  const byDay = {}
  interviews.forEach(iv => {
    const d = new Date(iv.date + 'T00:00:00')
    if (d.getFullYear() === year && d.getMonth() === month) {
      const day = d.getDate()
      if (!byDay[day]) byDay[day] = []
      byDay[day].push(iv)
    }
  })

  const prevMonth = () => {
    if (month === 0) { setMonth(11); setYear(y => y - 1) }
    else setMonth(m => m - 1)
  }
  const nextMonth = () => {
    if (month === 11) { setMonth(0); setYear(y => y + 1) }
    else setMonth(m => m + 1)
  }

  const handleDelete = async (id) => {
    try {
      await deleteInterview(id)
      setInterviews(prev => prev.filter(i => i.interview_id !== id))
      setToast({ message: 'Entretien supprimé', type: 'info' })
      setSelectedDay(null)
    } catch {
      setToast({ message: 'Erreur lors de la suppression', type: 'error' })
    }
  }

  // Upcoming = les prochains entretiens triés par date
  const upcoming = [...interviews]
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(0, 6)

  return (
    <main className="p-10 flex-1">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Calendrier */}
        <div className="col-span-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-5xl font-black text-on-surface tracking-tight">
                {MONTHS_FR[month]} {year}
              </h1>
              <p className="text-on-surface-variant font-medium mt-1">
                {loading ? '...' : `${interviews.length} entretien${interviews.length > 1 ? 's' : ''} ce mois`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={prevMonth}
                className="w-10 h-10 rounded-xl hover:bg-surface-container-high flex items-center justify-center text-on-surface-variant transition-colors">
                <Icon name="chevron_left" size={22} />
              </button>
              <button onClick={() => { setYear(now.getFullYear()); setMonth(now.getMonth()) }}
                className="px-4 py-2 rounded-xl bg-primary text-white text-sm font-bold hover:opacity-90 transition-opacity">
                Aujourd'hui
              </button>
              <button onClick={nextMonth}
                className="w-10 h-10 rounded-xl hover:bg-surface-container-high flex items-center justify-center text-on-surface-variant transition-colors">
                <Icon name="chevron_right" size={22} />
              </button>
            </div>
          </div>

          {/* En-têtes jours */}
          <div className="grid grid-cols-7 mb-2">
            {DAYS_OF_WEEK.map(d => (
              <div key={d} className="text-center text-[11px] font-bold text-outline uppercase tracking-widest py-2">{d}</div>
            ))}
          </div>

          {/* Cellules */}
          <div className="grid grid-cols-7 gap-1">
            {cells.map((day, i) => {
              const events  = day ? (byDay[day] || []) : []
              const isToday = isThisMonth && day === today
              const isSelected = selectedDay === day
              return (
                <div key={i}
                  onClick={() => day && setSelectedDay(isSelected ? null : day)}
                  className={`min-h-[100px] rounded-2xl p-2 transition-colors cursor-pointer
                    ${!day ? '' : isToday
                      ? 'bg-primary/5 ring-2 ring-primary/20'
                      : isSelected
                        ? 'bg-surface-container ring-2 ring-primary/30'
                        : 'hover:bg-surface-container-low/50'}`}>
                  {day && (
                    <>
                      <span className={`text-sm font-bold w-7 h-7 flex items-center justify-center rounded-full mb-1
                        ${isToday ? 'bg-primary text-white' : 'text-on-surface-variant'}`}>
                        {day}
                      </span>
                      <div className="space-y-1">
                        {events.slice(0, 2).map((iv) => (
                          <div key={iv.interview_id}
                            onClick={e => { e.stopPropagation(); onNavigate?.('profile', iv.candidate_id) }}
                            className={`text-[10px] font-bold px-2 py-1 rounded-lg truncate cursor-pointer hover:opacity-80 transition-opacity ${typeColor(iv.type)}`}>
                            {initials(iv.candidate_name)} · {iv.time ?? ''}
                          </div>
                        ))}
                        {events.length > 2 && (
                          <div className="text-[10px] text-on-surface-variant font-medium px-2">+{events.length - 2}</div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )
            })}
          </div>

          {/* Détail jour sélectionné */}
          {selectedDay && byDay[selectedDay] && (
            <div className="mt-6 bg-surface-container-lowest rounded-2xl p-6 shadow-ambient">
              <h3 className="text-base font-extrabold text-on-surface mb-4">
                Entretiens du {selectedDay} {MONTHS_FR[month]}
              </h3>
              <div className="space-y-3">
                {byDay[selectedDay].map(iv => (
                  <div key={iv.interview_id} className="flex items-center gap-4 p-3 bg-surface-container-low rounded-xl group">
                    <div className={`px-3 py-1 rounded-lg text-xs font-bold flex-shrink-0 ${typeColor(iv.type)}`}>
                      {iv.time ?? '—'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-bold text-on-surface truncate">{iv.candidate_name ?? 'Anonyme'}</p>
                      <p className="text-xs text-on-surface-variant">{iv.type ?? '—'}</p>
                      {iv.notes && <p className="text-xs text-on-surface-variant/60 truncate mt-0.5">{iv.notes}</p>}
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => onNavigate?.('profile', iv.candidate_id)}
                        className="p-1.5 hover:bg-surface-container-high rounded-lg text-on-surface-variant"
                        title="Voir le profil">
                        <Icon name="person" size={16} />
                      </button>
                      <button onClick={() => handleDelete(iv.interview_id)}
                        className="p-1.5 hover:bg-error-container rounded-lg text-on-surface-variant hover:text-error transition-colors"
                        title="Annuler l'entretien">
                        <Icon name="event_busy" size={16} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Légende */}
          <div className="mt-4 flex items-center gap-6 flex-wrap">
            {Object.entries(TYPE_COLORS).slice(0, 4).map(([type, cls]) => (
              <div key={type} className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${cls.split(' ')[0]}`} />
                <span className="text-xs font-medium text-on-surface-variant">{type}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="col-span-4 space-y-6">
          {/* Prochains entretiens */}
          <div className="bg-surface-container-lowest rounded-2xl p-6 shadow-ambient">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-base font-extrabold text-on-surface">Prochains entretiens</h3>
              <span className="text-xs font-bold text-primary bg-primary/10 px-2 py-1 rounded-full">
                {interviews.length}
              </span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center py-8 text-on-surface-variant gap-2">
                <Icon name="hourglass_empty" size={20} />
                <span className="text-sm">Chargement...</span>
              </div>
            ) : upcoming.length === 0 ? (
              <div className="text-center py-8 text-on-surface-variant">
                <Icon name="event_busy" size={32} />
                <p className="text-sm font-medium mt-2">Aucun entretien planifié</p>
                <p className="text-xs mt-1 opacity-60">Planifiez depuis le profil candidat</p>
              </div>
            ) : (
              <div className="space-y-3">
                {upcoming.map((iv) => {
                  const d = new Date(iv.date + 'T00:00:00')
                  const dateStr = d.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' })
                  return (
                    <div key={iv.interview_id}
                      onClick={() => onNavigate?.('profile', iv.candidate_id)}
                      className="flex items-center gap-3 p-3 rounded-xl hover:bg-surface-container-low transition-colors cursor-pointer group">
                      <div className="flex-shrink-0 text-center">
                        <p className="text-xs font-black text-primary leading-none">{d.getDate()}</p>
                        <p className="text-[10px] font-medium text-on-surface-variant uppercase">{MONTHS_FR[d.getMonth()].slice(0,3)}</p>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold text-on-surface truncate group-hover:text-primary transition-colors">
                          {iv.candidate_name ?? 'Anonyme'}
                        </p>
                        <p className="text-[11px] text-on-surface-variant">{iv.type ?? '—'} · {iv.time ?? '—'}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* IA Insight */}
          <div className="bg-gradient-to-br from-primary to-primary-container p-6 rounded-2xl text-white shadow-lg relative overflow-hidden">
            <div className="relative z-10">
              <Icon name="auto_awesome" fill size={28} />
              <h4 className="text-base font-bold mt-3 mb-2">Insight IA</h4>
              <p className="text-sm opacity-90 leading-relaxed">
                {interviews.length > 0
                  ? `${interviews.length} entretien${interviews.length > 1 ? 's' : ''} planifié${interviews.length > 1 ? 's' : ''} ce mois. Cliquez sur un jour pour voir les détails.`
                  : 'Aucun entretien ce mois. Planifiez depuis le profil d\'un candidat invité.'}
              </p>
            </div>
            <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-white/10 rounded-full blur-2xl" />
          </div>
        </div>
      </div>
    </main>
  )
}
