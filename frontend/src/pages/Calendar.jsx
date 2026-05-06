import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getCandidates } from '../lib/api'

const DAYS_OF_WEEK = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
const MONTHS_FR = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']

const AVATAR_COLORS = [
  'bg-blue-100 text-primary',
  'bg-secondary-fixed text-secondary',
  'bg-green-100 text-tertiary',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

function initials(name) {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

export default function Calendar({ onNavigate }) {
  const now   = new Date()
  const [year, setYear]   = useState(now.getFullYear())
  const [month, setMonth] = useState(now.getMonth()) // 0-indexed
  const [invited, setInvited] = useState([])

  useEffect(() => {
    getCandidates({ decision: 'invite', limit: 50 })
      .then(setInvited)
      .catch(() => setInvited([]))
  }, [])

  const today        = now.getDate()
  const isThisMonth  = year === now.getFullYear() && month === now.getMonth()

  const firstDay    = new Date(year, month, 1).getDay() // 0=Sun
  // Convert Sunday-based to Monday-based
  const leadingBlanks = (firstDay + 6) % 7
  const daysInMonth  = new Date(year, month + 1, 0).getDate()

  const cells = [
    ...Array(leadingBlanks).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ]
  while (cells.length % 7 !== 0) cells.push(null)

  // Distribue les candidats invités sur des jours du mois (basé sur received_at ou index)
  const candidatesByDay = {}
  invited.forEach((c, i) => {
    let day
    if (c.received_at) {
      const d = new Date(c.received_at)
      if (d.getFullYear() === year && d.getMonth() === month) {
        day = d.getDate()
      }
    }
    if (!day) {
      // Distribue uniformément dans le mois si pas de date ce mois
      day = ((i * 7 + 3) % daysInMonth) + 1
    }
    if (!candidatesByDay[day]) candidatesByDay[day] = []
    candidatesByDay[day].push(c)
  })

  const prevMonth = () => {
    if (month === 0) { setMonth(11); setYear(y => y - 1) }
    else setMonth(m => m - 1)
  }
  const nextMonth = () => {
    if (month === 11) { setMonth(0); setYear(y => y + 1) }
    else setMonth(m => m + 1)
  }

  // Upcoming = tous les candidats invités triés par received_at
  const upcoming = [...invited].sort((a, b) => new Date(b.received_at) - new Date(a.received_at)).slice(0, 5)

  return (
    <main className="p-10 flex-1">
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Calendar */}
        <div className="col-span-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-5xl font-black text-on-surface tracking-tight">
                {MONTHS_FR[month]} {year}
              </h1>
              <p className="text-on-surface-variant font-medium mt-1">
                {invited.length} candidat{invited.length > 1 ? 's' : ''} invité{invited.length > 1 ? 's' : ''}
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

          {/* Grid header */}
          <div className="grid grid-cols-7 mb-2">
            {DAYS_OF_WEEK.map(d => (
              <div key={d} className="text-center text-[11px] font-bold text-outline uppercase tracking-widest py-2">{d}</div>
            ))}
          </div>

          {/* Grid cells */}
          <div className="grid grid-cols-7 gap-1">
            {cells.map((day, i) => {
              const events = day ? (candidatesByDay[day] || []) : []
              const isToday = isThisMonth && day === today
              return (
                <div key={i} className={`min-h-[100px] rounded-2xl p-2 transition-colors
                  ${!day ? '' : isToday ? 'bg-primary/5 ring-2 ring-primary/20' : 'hover:bg-surface-container-low/50'}`}>
                  {day && (
                    <>
                      <span className={`text-sm font-bold w-7 h-7 flex items-center justify-center rounded-full mb-1
                        ${isToday ? 'bg-primary text-white' : 'text-on-surface-variant'}`}>
                        {day}
                      </span>
                      <div className="space-y-1">
                        {events.slice(0, 2).map((c, j) => (
                          <div key={j}
                            onClick={() => onNavigate?.('profile', c.candidate_id)}
                            className="text-[10px] font-bold px-2 py-1 rounded-lg bg-primary/10 text-primary truncate cursor-pointer hover:bg-primary/20 transition-colors">
                            {initials(c.name)} · {Math.round((c.score ?? 0) * 100)}%
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

          {/* Legend */}
          <div className="mt-6 flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-primary" />
              <span className="text-xs font-medium text-on-surface-variant">Candidat invité</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-primary/5 ring-2 ring-primary/30" />
              <span className="text-xs font-medium text-on-surface-variant">Aujourd'hui</span>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="col-span-4 space-y-6">
          {/* Candidats invités */}
          <div className="bg-surface-container-lowest rounded-2xl p-6 shadow-ambient">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-base font-extrabold text-on-surface">Candidats invités</h3>
              <span className="text-xs font-bold text-primary bg-primary/10 px-2 py-1 rounded-full">{invited.length}</span>
            </div>

            {invited.length === 0 ? (
              <div className="text-center py-8 text-on-surface-variant">
                <Icon name="event_busy" size={32} />
                <p className="text-sm font-medium mt-2">Aucun candidat invité</p>
              </div>
            ) : (
              <div className="space-y-3">
                {upcoming.map((c, i) => {
                  const pct = Math.round((c.score ?? 0) * 100)
                  const date = c.received_at
                    ? new Date(c.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' })
                    : '—'
                  return (
                    <div key={c.candidate_id}
                      onClick={() => onNavigate?.('profile', c.candidate_id)}
                      className="flex items-center gap-3 p-3 rounded-xl hover:bg-surface-container-low transition-colors cursor-pointer group">
                      <div className={`w-10 h-10 rounded-xl ${AVATAR_COLORS[i % AVATAR_COLORS.length]} flex items-center justify-center font-bold text-xs flex-shrink-0`}>
                        {initials(c.name)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold text-on-surface truncate group-hover:text-primary transition-colors">
                          {c.name ?? 'Anonyme'}
                        </p>
                        <p className="text-[11px] text-on-surface-variant">{c.sector ?? '—'} · {date}</p>
                      </div>
                      <span className="text-xs font-black text-tertiary bg-tertiary/10 px-2 py-1 rounded-lg flex-shrink-0">
                        {pct}%
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* AI Insight */}
          <div className="bg-gradient-to-br from-primary to-primary-container p-6 rounded-2xl text-white shadow-lg relative overflow-hidden">
            <div className="relative z-10">
              <Icon name="auto_awesome" fill size={28} />
              <h4 className="text-base font-bold mt-3 mb-2">Insight IA</h4>
              <p className="text-sm opacity-90 leading-relaxed">
                {invited.length > 0
                  ? `${invited.length} candidat${invited.length > 1 ? 's' : ''} invité${invited.length > 1 ? 's' : ''} à contacter pour planifier les entretiens.`
                  : 'Aucun candidat invité pour le moment.'}
              </p>
            </div>
            <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-white/10 rounded-full blur-2xl" />
          </div>
        </div>
      </div>
    </main>
  )
}
