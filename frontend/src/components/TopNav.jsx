import { useState } from 'react'
import { Icon } from './Icon'

const HELP_ARTICLES = [
  { title: 'Comment envoyer un CV par email ?', desc: 'Envoyez un CV à 73cn1.test@inbox.testmail.app' },
  { title: 'Comprendre le score ML', desc: 'Le score indique la probabilité de correspondance au poste' },
  { title: 'Inviter un candidat', desc: 'Cliquez sur un candidat et sélectionnez "Planifier l\'entretien"' },
  { title: 'Supprimer un candidat', desc: 'Utilisez l\'icône corbeille dans la liste des candidats' },
]

export default function TopNav({ onNavigate }) {
  const [showNotif, setShowNotif]   = useState(false)
  const [showHelp,  setShowHelp]    = useState(false)

  return (
    <header className="bg-surface/80 backdrop-blur-xl sticky top-0 z-40 flex justify-between items-center w-full px-8 py-3">
      {/* Search */}
      <div className="flex items-center gap-8 flex-1">
        <div className="relative w-full max-w-md">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-outline">
            <Icon name="search" size={18} />
          </span>
          <input
            className="w-full bg-surface-container-high border-none rounded-xl py-2 pl-10 pr-4 text-sm outline-none focus:ring-2 focus:ring-primary/30 transition-all"
            placeholder="Rechercher candidats, offres..."
          />
        </div>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-4 relative">

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => { setShowNotif(v => !v); setShowHelp(false) }}
            className="p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
            <Icon name="notifications" size={22} />
          </button>
          {showNotif && (
            <div className="absolute right-0 top-12 w-80 bg-surface-container-lowest rounded-2xl shadow-ambient-lg z-50 overflow-hidden">
              <div className="px-5 py-4 border-b border-outline-variant/20">
                <h3 className="font-bold text-on-surface text-sm">Notifications</h3>
              </div>
              <div className="p-4 text-center text-on-surface-variant py-8">
                <Icon name="notifications_none" size={32} />
                <p className="text-sm font-medium mt-2">Aucune notification</p>
                <p className="text-xs mt-1 opacity-60">Les nouveaux CVs apparaîtront ici</p>
              </div>
            </div>
          )}
        </div>

        {/* Help */}
        <div className="relative">
          <button
            onClick={() => { setShowHelp(v => !v); setShowNotif(false) }}
            className="p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
            <Icon name="help_outline" size={22} />
          </button>
          {showHelp && (
            <div className="absolute right-0 top-12 w-80 bg-surface-container-lowest rounded-2xl shadow-ambient-lg z-50 overflow-hidden">
              <div className="px-5 py-4 border-b border-outline-variant/20">
                <h3 className="font-bold text-on-surface text-sm">Centre d'aide</h3>
              </div>
              <div className="divide-y divide-outline-variant/10">
                {HELP_ARTICLES.map((a, i) => (
                  <div key={i} className="px-5 py-4 hover:bg-surface-container-low transition-colors cursor-pointer">
                    <p className="text-sm font-bold text-on-surface">{a.title}</p>
                    <p className="text-xs text-on-surface-variant mt-0.5">{a.desc}</p>
                  </div>
                ))}
              </div>
              <div className="px-5 py-3 bg-surface-container-low/50">
                <p className="text-xs text-on-surface-variant text-center">Luminary ATS v1.0 · <span className="text-primary font-medium">api.lony.app</span></p>
              </div>
            </div>
          )}
        </div>

        <div className="h-8 w-px bg-outline-variant/30" />

        <button
          onClick={() => onNavigate('jobs')}
          className="px-5 py-2 bg-gradient-to-r from-primary to-primary-container text-white text-sm font-semibold rounded-full shadow-sm hover:shadow-md transition-all active:scale-95">
          Créer une offre
        </button>

        <button onClick={() => onNavigate('settings')}
          className="w-9 h-9 rounded-full bg-secondary-container flex items-center justify-center text-primary font-bold text-sm border-2 border-white shadow-sm select-none hover:opacity-80 transition-opacity">
          HR
        </button>
      </div>
    </header>
  )
}
