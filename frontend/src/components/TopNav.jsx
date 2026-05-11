import { useState, useCallback } from 'react'
import { Icon } from './Icon'
import { getCandidates } from '../lib/api'
import { useAuth } from '../contexts/AuthContext'

const HELP_ARTICLES = [
  { title: 'Comment envoyer un CV par email ?', desc: 'Envoyez un CV à 73cn1.test@inbox.testmail.app' },
  { title: 'Comprendre le score IA', desc: 'Le score indique la correspondance au poste cible' },
  { title: 'Inviter un candidat', desc: "Ouvrez le profil et cliquez sur « Planifier l'entretien »" },
  { title: 'Archiver un candidat', desc: 'Utilisez le menu Actions dans la liste des candidats' },
]

export default function TopNav({ onNavigate }) {
  const { user } = useAuth()
  const [showNotif,  setShowNotif]  = useState(false)
  const [showHelp,   setShowHelp]   = useState(false)
  const [searchQ,    setSearchQ]    = useState('')
  const [searchRes,  setSearchRes]  = useState([])
  const [searching,  setSearching]  = useState(false)
  const [showSearch, setShowSearch] = useState(false)

  const handleSearch = useCallback(async (val) => {
    setSearchQ(val)
    if (!val.trim()) { setSearchRes([]); setShowSearch(false); return }
    setSearching(true)
    setShowSearch(true)
    try {
      const res = await getCandidates({ q: val, limit: 5 })
      setSearchRes(res)
    } catch (_) { setSearchRes([]) }
    finally { setSearching(false) }
  }, [])

  const close = () => { setShowNotif(false); setShowHelp(false) }

  return (
    <header className="bg-card border-b border-border sticky top-0 z-40 flex items-center justify-between px-8 py-3 gap-6">

      {/* Search */}
      <div className="relative w-full max-w-sm">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
          <Icon name="search" size={17} />
        </span>
        <input
          value={searchQ}
          onChange={e => handleSearch(e.target.value)}
          onBlur={() => setTimeout(() => setShowSearch(false), 200)}
          onFocus={() => searchQ && setShowSearch(true)}
          className="w-full bg-muted border border-border rounded-lg py-2 pl-9 pr-4 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:ring-2 focus:ring-primary/20 transition-all"
          placeholder="Rechercher candidats, offres…"
        />
        {showSearch && (
          <div className="absolute top-11 left-0 right-0 bg-card border border-border rounded-xl shadow-card-lg z-50 overflow-hidden">
            {searching ? (
              <p className="px-4 py-3 text-sm text-muted-foreground">Recherche…</p>
            ) : searchRes.length === 0 ? (
              <p className="px-4 py-3 text-sm text-muted-foreground">Aucun résultat</p>
            ) : (
              <div className="divide-y divide-border">
                {searchRes.map(c => (
                  <button
                    key={c.candidate_id}
                    onMouseDown={() => { onNavigate('profile', c.candidate_id); setSearchQ(''); setShowSearch(false) }}
                    className="w-full text-left px-4 py-3 hover:bg-muted transition-colors flex items-center gap-3"
                  >
                    <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center text-xs font-bold text-foreground flex-shrink-0">
                      {(c.name ?? '??').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-foreground">{c.name ?? 'Anonyme'}</p>
                      <p className="text-xs text-muted-foreground">{c.sector ?? '—'} · {Math.round((c.score ?? 0) * 100)}%</p>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right */}
      <div className="flex items-center gap-2">

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => { setShowNotif(v => !v); setShowHelp(false) }}
            className="w-9 h-9 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <Icon name="notifications" size={20} />
          </button>
          {showNotif && (
            <div className="absolute right-0 top-11 w-80 bg-card border border-border rounded-xl shadow-card-lg z-50 overflow-hidden">
              <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                <p className="text-sm font-semibold text-foreground">Notifications</p>
                <button onClick={close}><Icon name="close" size={16} /></button>
              </div>
              <div className="flex flex-col items-center justify-center py-10 gap-2 text-muted-foreground">
                <Icon name="notifications_none" size={32} />
                <p className="text-sm font-medium">Aucune notification</p>
                <p className="text-xs opacity-60">Les nouveaux CVs apparaîtront ici</p>
              </div>
            </div>
          )}
        </div>

        {/* Help */}
        <div className="relative">
          <button
            onClick={() => { setShowHelp(v => !v); setShowNotif(false) }}
            className="w-9 h-9 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <Icon name="help_outline" size={20} />
          </button>
          {showHelp && (
            <div className="absolute right-0 top-11 w-80 bg-card border border-border rounded-xl shadow-card-lg z-50 overflow-hidden">
              <div className="px-4 py-3 border-b border-border flex items-center justify-between">
                <p className="text-sm font-semibold text-foreground">Centre d'aide</p>
                <button onClick={close}><Icon name="close" size={16} /></button>
              </div>
              <div className="divide-y divide-border">
                {HELP_ARTICLES.map((a, i) => (
                  <div key={i} className="px-4 py-3 hover:bg-muted transition-colors cursor-pointer">
                    <p className="text-sm font-semibold text-foreground">{a.title}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{a.desc}</p>
                  </div>
                ))}
              </div>
              <div className="px-4 py-2.5 bg-muted/50 border-t border-border">
                <p className="text-xs text-muted-foreground text-center">Luminary ATS · api.lony.app</p>
              </div>
            </div>
          )}
        </div>

        <div className="h-6 w-px bg-border mx-1" />

        <button
          onClick={() => onNavigate('jobs')}
          className="px-4 py-2 bg-foreground text-background text-sm font-semibold rounded-lg hover:opacity-90 transition-opacity"
        >
          + Créer une offre
        </button>

        <button
          onClick={() => onNavigate('settings')}
          title={user?.name ?? 'Profil'}
          className="w-9 h-9 rounded-full bg-secondary border border-border flex items-center justify-center text-foreground font-bold text-xs hover:opacity-80 transition-opacity select-none"
        >
          {(user?.name ?? 'HR').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
        </button>
      </div>
    </header>
  )
}
