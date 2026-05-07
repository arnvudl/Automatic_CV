import { Icon } from './Icon'
import { useAuth } from '../contexts/AuthContext'

const NAV_ITEMS = [
  { id: 'dashboard',  label: 'Tableau de bord', icon: 'dashboard' },
  { id: 'jobs',       label: 'Offres',           icon: 'work_outline' },
  { id: 'candidates', label: 'Candidats',        icon: 'group' },
  { id: 'archives',   label: 'Archives',         icon: 'inventory_2' },
  { id: 'calendar',   label: 'Entretiens',       icon: 'calendar_today' },
  { id: 'settings',   label: 'Paramètres',       icon: 'settings' },
]

export default function Sidebar({ active, onNavigate }) {
  const { logout, user } = useAuth()
  return (
    <aside className="h-screen w-64 fixed left-0 top-0 z-50 bg-surface-container-low flex flex-col p-6 space-y-8">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/20">
          <Icon name="auto_awesome" fill size={20} />
        </div>
        <div>
          <h1 className="text-lg font-black text-primary leading-tight">Luminary ATS</h1>
          <p className="text-[10px] uppercase tracking-widest text-outline font-bold">Recruitment Suite</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 flex flex-col space-y-1">
        {NAV_ITEMS.map(({ id, label, icon }) => {
          const isActive = active === id
          return (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 w-full text-left
                ${isActive
                  ? 'bg-surface-container-lowest text-primary shadow-sm translate-x-1'
                  : 'text-slate-600 hover:text-primary hover:bg-white/50'}`}
            >
              <Icon name={icon} fill={isActive} size={20} />
              {label}
            </button>
          )
        })}
      </nav>

      {/* Bottom */}
      <div className="flex flex-col space-y-1 border-t border-outline-variant/20 pt-6">
        <a href="https://docs.lony.app" target="_blank" rel="noopener noreferrer"
          className="flex items-center gap-3 px-4 py-2 text-slate-600 hover:text-primary transition-all text-sm">
          <Icon name="help" size={20} />
          <span>Centre d'aide</span>
        </a>
        {user && (
          <div className="flex items-center gap-3 px-4 py-2 text-slate-600 text-sm">
            <div className="w-7 h-7 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold text-xs">
              {(user.name ?? 'U').split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase()}
            </div>
            <span className="font-medium truncate">{user.name ?? user.email}</span>
          </div>
        )}
        <button onClick={logout}
          className="flex items-center gap-3 px-4 py-2 text-slate-600 hover:text-error transition-all text-sm">
          <Icon name="logout" size={20} />
          <span>Déconnexion</span>
        </button>
      </div>
    </aside>
  )
}
