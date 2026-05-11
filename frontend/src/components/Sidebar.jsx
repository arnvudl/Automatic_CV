import { Icon } from './Icon'
import { useAuth } from '../contexts/AuthContext'

const MAIN_NAV = [
  { id: 'dashboard',  label: 'Tableau de bord', icon: 'dashboard' },
  { id: 'jobs',       label: "Offres d'emploi", icon: 'work_outline' },
  { id: 'pipeline',   label: 'Pipeline',         icon: 'view_kanban' },
  { id: 'candidates', label: 'Candidats',        icon: 'group' },
  { id: 'archives',   label: 'Archives',         icon: 'inventory_2' },
  { id: 'calendar',   label: 'Entretiens',       icon: 'calendar_today' },
]

const OTHER_NAV = [
  { id: 'settings', label: 'Paramètres', icon: 'settings' },
]

function NavItem({ id, label, icon, active, onClick }) {
  const isActive = active === id
  return (
    <button
      onClick={() => onClick(id)}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150
        ${isActive
          ? 'bg-white/10 text-white'
          : 'text-sidebar-muted hover:bg-white/5 hover:text-white'}`}
    >
      <Icon name={icon} fill={isActive} size={18} />
      {label}
    </button>
  )
}

export default function Sidebar({ active, onNavigate }) {
  const { logout, user } = useAuth()
  const initials = (name = '') =>
    (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || 'HR'

  return (
    <aside className="h-screen w-64 fixed left-0 top-0 z-50 bg-sidebar flex flex-col">

      {/* Logo */}
      <div className="px-5 py-5 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0">
            <Icon name="auto_awesome" fill size={16} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">Luminary ATS</p>
            <p className="text-[10px] text-sidebar-muted uppercase tracking-widest">Recrutement</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <p className="px-3 pb-2 text-[10px] font-semibold text-sidebar-muted uppercase tracking-widest">
          Principal
        </p>
        {MAIN_NAV.map(item => (
          <NavItem key={item.id} {...item} active={active} onClick={onNavigate} />
        ))}

        <p className="px-3 pt-5 pb-2 text-[10px] font-semibold text-sidebar-muted uppercase tracking-widest">
          Autre
        </p>
        {OTHER_NAV.map(item => (
          <NavItem key={item.id} {...item} active={active} onClick={onNavigate} />
        ))}
      </nav>

      {/* Support card */}
      <div className="px-3 pb-3">
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <div className="flex items-start justify-between mb-1">
            <p className="text-xs font-semibold text-white">Besoin d'aide ?</p>
            <Icon name="close" size={14} className="text-sidebar-muted cursor-pointer" />
          </div>
          <p className="text-[11px] text-sidebar-muted leading-relaxed mb-3">
            Contactez notre équipe pour toute question.
          </p>
          <a
            href="https://docs.lony.app"
            target="_blank"
            rel="noopener noreferrer"
            className="block text-center text-[11px] font-semibold bg-white/10 hover:bg-white/15 text-white px-3 py-1.5 rounded-lg transition-colors"
          >
            Nous contacter
          </a>
        </div>
      </div>

      {/* User */}
      <div className="px-3 pb-4 border-t border-sidebar-border pt-3">
        <div className="flex items-center gap-3 px-2 py-1.5">
          <div className="w-8 h-8 rounded-full bg-white/15 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
            {initials(user?.name)}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-white truncate">
              {user?.name ?? user?.email ?? 'Utilisateur'}
            </p>
            <p className="text-[10px] text-sidebar-muted truncate">{user?.email ?? ''}</p>
          </div>
          <button
            onClick={logout}
            title="Déconnexion"
            className="text-sidebar-muted hover:text-white transition-colors flex-shrink-0"
          >
            <Icon name="logout" size={16} />
          </button>
        </div>
      </div>
    </aside>
  )
}
