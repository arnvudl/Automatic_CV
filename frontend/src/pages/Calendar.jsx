import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getInterviews, deleteInterview } from '../lib/api'
import { Toast } from '../components/Toast'

const DAYS_OF_WEEK = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
const MONTHS_FR    = ['Janvier','Février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre']

const TYPE_COLORS = {
  'Entretien RH':        'bg-foreground/10 text-foreground',
  'Entretien technique': 'bg-success/10 text-success',
  'Entretien final':     'bg-purple-100 text-purple-700',
  'Présentation équipe': 'bg-amber-100 text-amber-700',
  'Autre':               'bg-muted text-muted-foreground',
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

  const upcoming = [...interviews]
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(0, 6)

  return (
    <main className="p-10 flex-1">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Calendar */}
        <div className="col-span-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-foreground tracking-tight">
                {MONTHS_FR[month]} {year}
              </h1>
              <p className="text-muted-foreground text-sm mt-1">
                {loading ? '…' : `${interviews.length} entretien${interviews.length !== 1 ? 's' : ''} ce mois`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={prevMonth}
                className="w-9 h-9 rounded-lg hover:bg-muted flex items-center justify-center text-muted-foreground transition-colors">
                <Icon name="chevron_left" size={20} />
              </button>
              <button onClick={() => { setYear(now.getFullYear()); setMonth(now.getMonth()) }}
                className="px-4 py-2 rounded-lg bg-foreground text-primary-foreground text-sm font-semibold hover:opacity-90 transition-opacity">
                Aujourd'hui
              </button>
              <button onClick={nextMonth}
                className="w-9 h-9 rounded-lg hover:bg-muted flex items-center justify-center text-muted-foreground transition-colors">
                <Icon name="chevron_right" size={20} />
              </button>
            </div>
          </div>

          {/* Day headers */}
          <div className="grid grid-cols-7 mb-1">
            {DAYS_OF_WEEK.map(d => (
              <div key={d} className="text-center text-[11px] font-bold text-muted-foreground uppercase tracking-widest py-2">{d}</div>
            ))}
          </div>

          {/* Cells */}
          <div className="grid grid-cols-7 gap-1">
            {cells.map((day, i) => {
              const events    = day ? (byDay[day] || []) : []
              const isToday   = isThisMonth && day === today
              const isSelected = selectedDay === day
              return (
                <div key={i}
                  onClick={() => day && setSelectedDay(isSelected ? null : day)}
                  className={`min-h-[96px] rounded-lg p-2 transition-colors cursor-pointer
                    ${!day ? '' : isToday
                      ? 'bg-foreground/5 ring-2 ring-foreground/20'
                      : isSelected
                        ? 'bg-muted ring-2 ring-border'
                        : 'hover:bg-muted/50'}`}>
                  {day && (
                    <>
                      <span className={`text-sm font-bold w-7 h-7 flex items-center justify-center rounded-full mb-1
                        ${isToday ? 'bg-foreground text-primary-foreground' : 'text-muted-foreground'}`}>
                        {day}
                      </span>
                      <div className="space-y-0.5">
                        {events.slice(0, 2).map((iv) => (
                          <div key={iv.interview_id}
                            onClick={e => { e.stopPropagation(); onNavigate?.('profile', iv.candidate_id) }}
                            className={`text-[10px] font-bold px-2 py-0.5 rounded truncate cursor-pointer hover:opacity-80 transition-opacity ${typeColor(iv.type)}`}>
                            {initials(iv.candidate_name)} · {iv.time ?? ''}
                          </div>
                        ))}
                        {events.length > 2 && (
                          <div className="text-[10px] text-muted-foreground px-2">+{events.length - 2}</div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              )
            })}
          </div>

          {/* Selected day detail */}
          {selectedDay && byDay[selectedDay] && (
            <div className="mt-5 bg-card border border-border rounded-xl p-5 shadow-card">
              <h3 className="text-sm font-bold text-foreground mb-4">
                Entretiens du {selectedDay} {MONTHS_FR[month]}
              </h3>
              <div className="space-y-2">
                {byDay[selectedDay].map(iv => (
                  <div key={iv.interview_id} className="flex items-center gap-4 p-3 bg-muted rounded-lg group">
                    <div className={`px-3 py-1 rounded-lg text-xs font-bold flex-shrink-0 ${typeColor(iv.type)}`}>
                      {iv.time ?? '—'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-foreground truncate">{iv.candidate_name ?? 'Anonyme'}</p>
                      <p className="text-xs text-muted-foreground">{iv.type ?? '—'}</p>
                      {iv.notes && <p className="text-xs text-muted-foreground/60 truncate mt-0.5">{iv.notes}</p>}
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => onNavigate?.('profile', iv.candidate_id)}
                        className="p-1.5 hover:bg-card rounded-lg text-muted-foreground hover:text-foreground transition-colors">
                        <Icon name="person" size={15} />
                      </button>
                      <button onClick={() => handleDelete(iv.interview_id)}
                        className="p-1.5 hover:bg-destructive/10 rounded-lg text-muted-foreground hover:text-destructive transition-colors">
                        <Icon name="event_busy" size={15} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="mt-4 flex items-center gap-5 flex-wrap">
            {Object.entries(TYPE_COLORS).slice(0, 4).map(([type, cls]) => (
              <div key={type} className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${cls.split(' ')[0]}`} />
                <span className="text-xs text-muted-foreground">{type}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="col-span-4 space-y-5">
          {/* Upcoming interviews */}
          <div className="bg-card border border-border rounded-xl p-5 shadow-card">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-bold text-foreground">Prochains entretiens</h3>
              <span className="text-xs font-bold text-foreground bg-muted px-2 py-1 rounded-full">
                {interviews.length}
              </span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center py-8 text-muted-foreground gap-2">
                <div className="w-4 h-4 border-2 border-border border-t-foreground rounded-full animate-spin" />
              </div>
            ) : upcoming.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Icon name="event_busy" size={28} />
                <p className="text-sm font-medium mt-2">Aucun entretien planifié</p>
                <p className="text-xs mt-1 opacity-60">Planifiez depuis le profil candidat</p>
              </div>
            ) : (
              <div className="space-y-2">
                {upcoming.map((iv) => {
                  const d = new Date(iv.date + 'T00:00:00')
                  return (
                    <div key={iv.interview_id}
                      onClick={() => onNavigate?.('profile', iv.candidate_id)}
                      className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors cursor-pointer group">
                      <div className="flex-shrink-0 text-center w-9">
                        <p className="text-sm font-black text-foreground leading-none">{d.getDate()}</p>
                        <p className="text-[10px] font-medium text-muted-foreground uppercase">{MONTHS_FR[d.getMonth()].slice(0, 3)}</p>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-foreground truncate group-hover:opacity-80 transition-opacity">
                          {iv.candidate_name ?? 'Anonyme'}
                        </p>
                        <p className="text-[11px] text-muted-foreground">{iv.type ?? '—'} · {iv.time ?? '—'}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* AI Insight */}
          <div className="bg-foreground p-5 rounded-xl text-primary-foreground relative overflow-hidden">
            <div className="relative z-10">
              <Icon name="auto_awesome" fill size={24} />
              <h4 className="text-sm font-bold mt-3 mb-2">Insight IA</h4>
              <p className="text-sm opacity-80 leading-relaxed">
                {interviews.length > 0
                  ? `${interviews.length} entretien${interviews.length !== 1 ? 's' : ''} planifié${interviews.length !== 1 ? 's' : ''} ce mois.`
                  : "Aucun entretien ce mois. Planifiez depuis le profil d'un candidat invité."}
              </p>
            </div>
            <div className="absolute -right-4 -bottom-4 w-20 h-20 bg-white/5 rounded-full blur-2xl" />
          </div>
        </div>
      </div>
    </main>
  )
}
