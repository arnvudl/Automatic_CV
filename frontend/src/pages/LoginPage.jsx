import { useState } from 'react'
import { Icon } from '../components/Icon'
import { useAuth } from '../contexts/AuthContext'

const BASE = import.meta.env.VITE_API_URL ?? ''

export default function LoginPage() {
  const { login }            = useAuth()
  const [email, setEmail]    = useState('')
  const [pass,  setPass]     = useState('')
  const [error, setError]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [showPass, setShowPass] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!email || !pass) return
    setError(null)
    setLoading(true)
    try {
      const res = await fetch(`${BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: pass }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data.detail ?? 'Email ou mot de passe incorrect')
        return
      }
      const { access_token, user } = await res.json()
      login(access_token, user)
    } catch {
      setError('Impossible de se connecter. Vérifiez votre connexion.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-foreground/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-foreground/5 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-sm relative">
        {/* Logo */}
        <div className="text-center mb-10">
          <div className="w-14 h-14 bg-foreground rounded-xl flex items-center justify-center text-primary-foreground shadow-card-md mx-auto mb-4">
            <Icon name="auto_awesome" fill size={28} />
          </div>
          <h1 className="text-2xl font-bold text-foreground tracking-tight">Luminary ATS</h1>
          <p className="text-xs text-muted-foreground mt-1 font-medium uppercase tracking-widest">Recruitment Suite</p>
        </div>

        {/* Card */}
        <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg">
          <h2 className="text-lg font-bold text-foreground mb-1">Connexion</h2>
          <p className="text-sm text-muted-foreground mb-7">Accès réservé à l'équipe RH</p>

          {error && (
            <div className="mb-5 flex items-center gap-3 p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm font-medium">
              <Icon name="error_outline" size={16} />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">
                Email
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                  <Icon name="mail" size={16} />
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="vous@entreprise.com"
                  className="w-full bg-muted border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 transition-all"
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2">
                Mot de passe
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                  <Icon name="lock" size={16} />
                </span>
                <input
                  type={showPass ? 'text' : 'password'}
                  value={pass}
                  onChange={e => setPass(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-muted border border-border rounded-lg pl-10 pr-12 py-2.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 transition-all"
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors">
                  <Icon name={showPass ? 'visibility_off' : 'visibility'} size={16} />
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !email || !pass}
              className="w-full py-2.5 rounded-lg bg-foreground text-primary-foreground font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50 flex items-center justify-center gap-2 mt-2">
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Connexion...
                </>
              ) : (
                <>
                  <Icon name="login" size={16} />
                  Se connecter
                </>
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-5 opacity-50">
          Luminary ATS v2.1 · Sécurisé JWT
        </p>
      </div>
    </div>
  )
}
