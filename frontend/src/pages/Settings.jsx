import { Icon } from '../components/Icon'

export default function Settings({ onNavigate }) {
  return (
    <div className="p-10 max-w-3xl mx-auto">
      <h1 className="text-5xl font-black text-on-surface tracking-tight mb-10">Paramètres</h1>

      <div className="space-y-6">
        {/* Profil RH */}
        <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
          <h2 className="text-lg font-extrabold text-on-surface mb-6 flex items-center gap-2">
            <Icon name="person" size={22} className="text-primary" /> Profil RH
          </h2>
          <div className="flex items-center gap-6 mb-6">
            <div className="w-20 h-20 rounded-full bg-primary/10 text-primary flex items-center justify-center text-2xl font-black">
              HR
            </div>
            <div>
              <p className="text-xl font-bold text-on-surface">Responsable RH</p>
              <p className="text-sm text-on-surface-variant">arnaudleroy20@gmail.com</p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'Nom', value: 'Responsable RH' },
              { label: 'Email', value: 'arnaudleroy20@gmail.com' },
              { label: 'Rôle', value: 'Recruteur' },
              { label: 'Organisation', value: 'Luminary ATS' },
            ].map(({ label, value }) => (
              <div key={label} className="p-4 bg-surface-container-low rounded-xl">
                <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">{label}</p>
                <p className="text-sm font-semibold text-on-surface">{value}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Système */}
        <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
          <h2 className="text-lg font-extrabold text-on-surface mb-6 flex items-center gap-2">
            <Icon name="settings" size={22} /> Système
          </h2>
          <div className="space-y-3">
            {[
              { icon: 'api', label: 'API', value: 'https://api.lony.app', color: 'text-tertiary' },
              { icon: 'cloud', label: 'Frontend', value: 'https://ats.lony.app', color: 'text-primary' },
              { icon: 'smart_toy', label: 'Modèle ML', value: 'SVM — Dernière mise à jour : 22 avril 2026', color: 'text-on-surface-variant' },
              { icon: 'database', label: 'Base de données', value: 'PostgreSQL 16 — Digital Ocean', color: 'text-on-surface-variant' },
            ].map(({ icon, label, value, color }) => (
              <div key={label} className="flex items-center gap-4 p-4 bg-surface-container-low rounded-xl">
                <span className={`${color}`}><Icon name={icon} size={20} /></span>
                <div>
                  <p className="text-xs font-bold text-on-surface-variant uppercase tracking-widest">{label}</p>
                  <p className="text-sm font-medium text-on-surface">{value}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Déconnexion */}
        <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
          <h2 className="text-lg font-extrabold text-on-surface mb-4 flex items-center gap-2">
            <Icon name="logout" size={22} className="text-error" /> Session
          </h2>
          <p className="text-sm text-on-surface-variant mb-6">
            Vous êtes connecté en tant que <strong>Responsable RH</strong>.
          </p>
          <button
            onClick={() => onNavigate('dashboard')}
            className="px-6 py-3 bg-error-container text-error font-bold rounded-xl hover:opacity-90 transition-opacity flex items-center gap-2 text-sm">
            <Icon name="logout" size={18} />
            Se déconnecter
          </button>
        </section>
      </div>
    </div>
  )
}
