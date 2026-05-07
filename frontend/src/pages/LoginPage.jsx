import { useState } from 'react'
import { Icon } from '../components/Icon'
import { useAuth } from '../contexts/AuthContext'

const BASE = import.meta.env.VITE_API_URL ?? ''

export default function LoginPage() {
  const { login }          = useAuth()
  const [email, setEmail]  = useState('')
  const [pass,  setPass]   = useState('')
  const [error, setError]  = useState(null)
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
    <div className="min-h-screen bg-surface flex items-center justify-center p-4">
      {/* Background blur orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-secondary/10 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-sm relative">
        {/* Logo */}
        <div className="text-center mb-10">
          <div className="w-16 h-16 bg-primary rounded-2xl flex items-center justify-center text-white shadow-xl shadow-primary/30 mx-auto mb-4">
            <Icon name="auto_awesome" fill size={32} />
          </div>
          <h1 className="text-3xl font-black text-primary tracking-tight">Luminary ATS</h1>
          <p className="text-sm text-on-surface-variant mt-1 font-medium uppercase tracking-widest">Recruitment Suite</p>
        </div>

        {/* Card */}
        <div className="bg-surface-container-lowest rounded-3xl p-8 shadow-ambient-lg">
          <h2 className="text-xl font-extrabold text-on-surface mb-1">Connexion</h2>
          <p className="text-sm text-on-surface-variant mb-8">Accès réservé à l'équipe RH</p>

          {error && (
            <div className="mb-5 flex items-center gap-3 p-4 bg-error-container rounded-xl text-error text-sm font-medium">
              <Icon name="error_outline" size={18} />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email */}
            <div>
              <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">
                Email
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-outline">
                  <Icon name="mail" size={18} />
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="vous@entreprise.com"
                  className="w-full bg-surface-container-low border-none rounded-xl pl-10 pr-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all"
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2">
                Mot de passe
              </label>
              <div className="relative">
                <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-outline">
                  <Icon name="lock" size={18} />
                </span>
                <input
                  type={showPass ? 'text' : 'password'}
                  value={pass}
                  onChange={e => setPass(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-surface-container-low border-none rounded-xl pl-10 pr-12 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all"
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-outline hover:text-on-surface transition-colors">
                  <Icon name={showPass ? 'visibility_off' : 'visibility'} size={18} />
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading || !email || !pass}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-lg shadow-primary/20 hover:shadow-xl active:scale-[0.98] transition-all disabled:opacity-50 flex items-center justify-center gap-2 mt-2">
              {loading ? (
                <>
                  <Icon name="hourglass_empty" size={18} />
                  Connexion...
                </>
              ) : (
                <>
                  <Icon name="login" size={18} />
                  Se connecter
                </>
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-on-surface-variant mt-6 opacity-50">
          Luminary ATS v2.1 · Sécurisé JWT
        </p>
      </div>
    </div>
  )
}
