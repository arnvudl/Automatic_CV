import { Icon } from '../components/Icon'

export default function Settings({ onNavigate }) {
  return (
    <div className="p-10 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold text-foreground tracking-tight mb-8">Paramètres</h1>

      <div className="space-y-5">
        {/* Profil RH */}
        <section className="bg-card border border-border rounded-xl p-8 shadow-card">
          <h2 className="text-base font-bold text-foreground mb-6 flex items-center gap-2">
            <Icon name="person" size={20} /> Profil RH
          </h2>
          <div className="flex items-center gap-5 mb-6">
            <div className="w-16 h-16 rounded-full bg-muted text-foreground flex items-center justify-center text-xl font-black">
              HR
            </div>
            <div>
              <p className="text-lg font-bold text-foreground">Responsable RH</p>
              <p className="text-sm text-muted-foreground">arnaudleroy20@gmail.com</p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Nom',          value: 'Responsable RH' },
              { label: 'Email',        value: 'arnaudleroy20@gmail.com' },
              { label: 'Rôle',         value: 'Recruteur' },
              { label: 'Organisation', value: 'Luminary ATS' },
            ].map(({ label, value }) => (
              <div key={label} className="p-4 bg-muted rounded-lg">
                <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">{label}</p>
                <p className="text-sm font-semibold text-foreground">{value}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Système */}
        <section className="bg-card border border-border rounded-xl p-8 shadow-card">
          <h2 className="text-base font-bold text-foreground mb-6 flex items-center gap-2">
            <Icon name="settings" size={20} /> Système
          </h2>
          <div className="space-y-2">
            {[
              { icon: 'api',        label: 'API',              value: 'https://api.lony.app',                             color: 'text-success' },
              { icon: 'cloud',      label: 'Frontend',         value: 'https://ats.lony.app',                             color: 'text-foreground' },
              { icon: 'smart_toy',  label: 'Modèle ML',        value: 'SVM — Dernière mise à jour : 22 avril 2026',       color: 'text-muted-foreground' },
              { icon: 'database',   label: 'Base de données',  value: 'PostgreSQL 16 — Digital Ocean',                    color: 'text-muted-foreground' },
            ].map(({ icon, label, value, color }) => (
              <div key={label} className="flex items-center gap-4 p-4 bg-muted rounded-lg">
                <span className={color}><Icon name={icon} size={18} /></span>
                <div>
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">{label}</p>
                  <p className="text-sm font-medium text-foreground">{value}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Session */}
        <section className="bg-card border border-border rounded-xl p-8 shadow-card">
          <h2 className="text-base font-bold text-foreground mb-4 flex items-center gap-2">
            <Icon name="logout" size={20} className="text-destructive" /> Session
          </h2>
          <p className="text-sm text-muted-foreground mb-5">
            Vous êtes connecté en tant que <strong className="text-foreground">Responsable RH</strong>.
          </p>
          <button
            onClick={() => onNavigate('dashboard')}
            className="px-5 py-2.5 bg-destructive/10 text-destructive font-bold rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2 text-sm">
            <Icon name="logout" size={16} />
            Se déconnecter
          </button>
        </section>
      </div>
    </div>
  )
}
