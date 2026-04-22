import { useState } from 'react'
import { Icon } from '../components/Icon'

const DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

// April 2026 — starts on Wednesday (index 2)
const EVENTS = {
  1:  [{ time: '10:00', label: 'Tech Interview', color: 'bg-blue-100 text-blue-700' }, { time: '14:30', label: 'HR Chat', color: 'bg-amber-100 text-amber-700' }],
  3:  [{ time: '09:15', label: 'Final Stage', color: 'bg-green-100 text-green-700' }],
  6:  [{ time: '11:00', label: 'Sarah Chen', color: 'bg-primary text-white' }, { time: '16:00', label: 'Panel Interview', color: 'bg-blue-100 text-blue-700' }],
  8:  [{ time: '13:00', label: 'Candidate Review', color: 'bg-amber-100 text-amber-700' }],
  14: [{ time: '09:00', label: 'Elena Rodriguez', color: 'bg-green-100 text-green-700' }],
  18: [{ time: '15:00', label: 'Group Assessment', color: 'bg-purple-100 text-purple-700' }],
  22: [{ time: '11:00', label: 'David Miller', color: 'bg-primary text-white' }],
}

const TODAY = 22

const UPCOMING = [
  { name: 'Sarah Chen', role: 'Senior Product Designer', time: '11:00 AM', type: 'Technical Interview', typeColor: 'bg-primary', initials: 'SC', color: 'bg-blue-100 text-primary' },
  { name: 'David Miller', role: 'Lead Data Analyst', time: '02:30 PM', type: 'HR Culture Fit', typeColor: 'bg-amber-500', typeTextColor: 'text-amber-600', initials: 'DM', color: 'bg-amber-100 text-amber-700' },
  { name: 'Elena Rodriguez', role: 'Backend Engineer', time: '04:00 PM', type: 'Final Interview', typeColor: 'bg-tertiary', typeTextColor: 'text-tertiary', initials: 'ER', color: 'bg-green-100 text-tertiary' },
]

const STAGE_LEGEND = [
  { icon: 'code', color: 'text-blue-700', label: 'Technical' },
  { icon: 'favorite', color: 'text-amber-700', label: 'HR / Cultural' },
  { icon: 'verified', color: 'text-green-700', label: 'Final Stage' },
  { icon: 'person_search', color: 'text-slate-400', label: 'Review' },
]

// April 2026 starts on Wednesday → 2 blank days before day 1
const LEADING_BLANKS = 2
const DAYS_IN_APRIL = 30

export default function Calendar() {
  const [month] = useState('April 2026')

  const cells = [
    ...Array(LEADING_BLANKS).fill(null),
    ...Array.from({ length: DAYS_IN_APRIL }, (_, i) => i + 1),
  ]

  // Pad to complete last row
  while (cells.length % 7 !== 0) cells.push(null)

  return (
    <main className="p-10 flex-1">
      <div className="grid grid-cols-12 gap-8 items-start">
        {/* Calendar */}
        <div className="col-span-12 xl:col-span-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-4xl font-extrabold tracking-tight text-on-surface">{month}</h2>
              <p className="text-on-surface-variant font-medium mt-1">12 scheduled interviews this month</p>
            </div>
            <div className="flex items-center gap-2 bg-surface-container-low p-1.5 rounded-full">
              <button className="p-2 hover:bg-white rounded-full transition-all text-slate-600"><Icon name="chevron_left" size={20} /></button>
              <button className="px-6 py-2 bg-white rounded-full text-sm font-bold shadow-sm">Today</button>
              <button className="p-2 hover:bg-white rounded-full transition-all text-slate-600"><Icon name="chevron_right" size={20} /></button>
            </div>
          </div>

          {/* Day headers */}
          <div className="grid grid-cols-7 mb-4">
            {DAYS_OF_WEEK.map(d => (
              <div key={d} className="text-center text-xs font-bold uppercase tracking-widest text-slate-400 py-4">{d}</div>
            ))}
          </div>

          {/* Calendar grid */}
          <div className="grid grid-cols-7 gap-px bg-outline-variant/10 rounded-3xl overflow-hidden border border-outline-variant/20 shadow-xl shadow-slate-200/50">
            {cells.map((day, i) => {
              const isToday = day === TODAY
              const isWeekend = (i + LEADING_BLANKS) % 7 >= 5 || i % 7 >= 5
              const events = day ? (EVENTS[day] || []) : []

              return (
                <div key={i} className={`min-h-[120px] p-3 relative group transition-colors
                  ${!day ? 'bg-surface-container-low/50' : isToday ? 'bg-primary/5 ring-2 ring-primary ring-inset' : 'bg-white hover:bg-surface-container-lowest'}
                  ${isWeekend && day ? 'bg-surface-container-lowest/50' : ''}`}>
                  {day && (
                    <>
                      <span className={`text-sm font-bold ${isToday ? 'text-primary' : 'text-on-surface'}`}>{day}</span>
                      <div className="mt-1.5 space-y-1">
                        {events.map((e, j) => (
                          <div key={j} className={`text-[10px] font-bold px-2 py-1 rounded-lg truncate ${e.color}`}>
                            {e.time} • {e.label}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Sidebar */}
        <div className="col-span-12 xl:col-span-4 space-y-8">
          {/* Upcoming Today */}
          <div className="bg-white rounded-[2.5rem] p-8 shadow-ambient-lg">
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-xl font-extrabold tracking-tight text-on-surface">Upcoming Today</h3>
              <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-xs font-bold">3 Events</span>
            </div>
            <div className="space-y-6">
              {UPCOMING.map((u, i) => (
                <div key={i} className={`flex gap-4 group cursor-pointer ${i < UPCOMING.length - 1 ? 'border-b border-surface-container-high pb-6' : ''}`}>
                  <div className={`w-14 h-14 rounded-2xl flex-shrink-0 flex items-center justify-center font-bold text-lg ${u.color}`}>
                    {u.initials}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h4 className="font-bold text-on-surface group-hover:text-primary transition-colors">{u.name}</h4>
                      <span className="text-xs font-bold text-slate-400">{u.time}</span>
                    </div>
                    <p className="text-sm text-on-surface-variant">{u.role}</p>
                    <div className="flex items-center mt-2 gap-2">
                      <span className={`w-2 h-2 rounded-full ${u.typeColor}`} />
                      <span className={`text-[10px] font-bold uppercase tracking-wider ${u.typeTextColor || 'text-primary'}`}>{u.type}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Stage Legend */}
          <div className="bg-surface-container-low rounded-[2.5rem] p-8">
            <h3 className="text-sm font-black uppercase tracking-[0.2em] text-slate-400 mb-6">Stage Legend</h3>
            <div className="grid grid-cols-2 gap-4">
              {STAGE_LEGEND.map(({ icon, color, label }) => (
                <div key={label} className="bg-white p-4 rounded-3xl flex flex-col gap-2 shadow-sm border border-slate-100">
                  <span className={color}><Icon name={icon} fill size={24} /></span>
                  <span className="text-xs font-bold text-on-surface">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* AI Insight */}
          <div className="bg-gradient-to-br from-primary to-primary-container rounded-[2.5rem] p-8 text-white shadow-2xl shadow-primary/20">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-blue-200"><Icon name="auto_awesome" size={20} /></span>
              <span className="text-xs font-bold uppercase tracking-widest text-blue-100">Luminary AI</span>
            </div>
            <p className="text-lg font-medium leading-relaxed mb-6 italic opacity-90">
              "Today's interview load is manageable. Recommended follow-up window: Tomorrow morning."
            </p>
            <button className="w-full bg-white/20 backdrop-blur-md text-white border border-white/30 py-3 rounded-full text-sm font-bold hover:bg-white/30 transition-all">
              View Analytics
            </button>
          </div>
        </div>
      </div>
    </main>
  )
}
