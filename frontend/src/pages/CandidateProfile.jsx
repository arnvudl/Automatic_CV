import { useState, useEffect, useCallback } from 'react'
import { Icon } from '../components/Icon'
import { getCandidate, updateStatus, createInterview } from '../lib/api'
import { Toast } from '../components/Toast'
import { InterviewModal } from '../components/InterviewModal'

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

function getScoreColors(score) {
  const pct = Math.round((score ?? 0) * 100)
  if (pct >= 75) return { text: 'text-tertiary', bg: 'bg-tertiary/10', bar: '#006b2b', label: 'Top Match' }
  if (pct >= 50) return { text: 'text-yellow-600', bg: 'bg-yellow-100', bar: '#ca8a04', label: 'Good Match' }
  return { text: 'text-error', bg: 'bg-error-container', bar: '#ba1a1a', label: 'Low Match' }
}

function ShapBar({ label, value }) {
  const abs = Math.abs(value)
  const width = Math.min((abs / 0.3) * 100, 100)
  const positive = value >= 0
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-36 text-on-surface-variant font-medium truncate">{label}</span>
      <div className="flex-1 h-2 bg-surface-container rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${positive ? 'bg-tertiary' : 'bg-error'}`} style={{ width: `${width}%` }} />
      </div>
      <span className={`w-12 text-right font-bold ${positive ? 'text-tertiary' : 'text-error'}`}>
        {positive ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  )
}

export default function CandidateProfile({ onNavigate, candidateId }) {
  const [candidate, setCandidate]   = useState(null)
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState(null)
  const [updating, setUpdating]     = useState(false)
  const [toast, setToast]           = useState(null)
  const [showInterview, setShowInterview] = useState(false)

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type })
  }, [])

  useEffect(() => {
    if (!candidateId) { setError('Candidat non spécifié'); setLoading(false); return }
    getCandidate(candidateId)
      .then(setCandidate)
      .catch(() => setError('Candidat introuvable'))
      .finally(() => setLoading(false))
  }, [candidateId])

  const handleStatus = async (newStatus) => {
    if (!candidate || updating) return
    setUpdating(true)
    try {
      await updateStatus(candidate.candidate_id, newStatus)
      setCandidate(prev => ({ ...prev, status: newStatus }))
      const messages = {
        archived:             'Candidat archivé',
        interview_scheduled:  'Entretien planifié ✓',
        review:               'Mis en révision',
        rejected:             'Candidat rejeté',
        inbox:                'Remis en boîte de réception',
      }
      showToast(messages[newStatus] ?? 'Statut mis à jour')
    } catch (e) {
      showToast('Erreur lors de la mise à jour', 'error')
    }
    finally { setUpdating(false) }
  }

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh] text-on-surface-variant gap-3">
      <Icon name="hourglass_empty" size={32} />
      <span className="font-medium">Chargement...</span>
    </div>
  )

  if (error || !candidate) return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-on-surface-variant">
      <Icon name="person_off" size={48} />
      <p className="font-bold text-on-surface">{error ?? 'Candidat introuvable'}</p>
      <button onClick={() => onNavigate('candidates')} className="text-primary font-bold text-sm hover:underline">
        Retour à la liste
      </button>
    </div>
  )

  const pct    = Math.round((candidate.score ?? 0) * 100)
  const colors = getScoreColors(candidate.score)
  const ini    = initials(candidate.name)
  const date   = candidate.received_at
    ? new Date(candidate.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'long', year: 'numeric' })
    : '—'

  const decisionLabel = { invite: 'Invité', reject: 'Rejeté', eliminated: 'Éliminé' }[candidate.decision] ?? '—'
  const decisionColor = { invite: 'text-tertiary', reject: 'text-error', eliminated: 'text-on-surface-variant' }[candidate.decision] ?? 'text-on-surface-variant'

  let shapEntries = []
  try {
    if (candidate.shap_json) {
      const shap = typeof candidate.shap_json === 'string' ? JSON.parse(candidate.shap_json) : candidate.shap_json
      shapEntries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6)
    }
  } catch (_) {}

  const infoFields = [
    { label: 'Email',           value: candidate.email },
    { label: 'Téléphone',       value: candidate.phone },
    { label: 'Secteur',         value: candidate.sector },
    { label: 'Rôle cible',      value: candidate.target_role },
    { label: 'Expérience',      value: candidate.years_experience != null ? `${candidate.years_experience} ans` : null },
    { label: "Niveau d'études", value: candidate.education_level },
    { label: 'Genre',           value: candidate.gender },
    { label: 'Âge',             value: candidate.age != null ? `${candidate.age} ans` : null },
    { label: 'Statut',          value: candidate.status },
    { label: 'Reçu le',         value: date },
  ].filter(f => f.value != null)

  return (
    <div className="p-10 max-w-7xl mx-auto">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
      {showInterview && (
        <InterviewModal
          candidate={candidate}
          onClose={() => setShowInterview(false)}
          onConfirm={async ({ date, time, type, notes }) => {
            setShowInterview(false)
            await handleStatus('interview_scheduled')
            try {
              await createInterview({
                candidate_id:   candidate.candidate_id,
                candidate_name: candidate.name,
                date, time, type, notes,
              })
            } catch (_) { /* non bloquant */ }
            showToast(`Entretien ${type} planifié le ${new Date(date).toLocaleDateString('fr-FR')} à ${time}`)
          }}
        />
      )}
      {/* Breadcrumb */}
      <div className="flex justify-between items-center mb-10">
        <button onClick={() => onNavigate('candidates')}
          className="flex items-center gap-2 text-on-surface-variant hover:text-on-surface transition-colors text-sm font-medium">
          <Icon name="arrow_back" size={18} />
          Retour aux candidats
        </button>
        <div className="flex gap-3 flex-wrap">
          {candidate.status === 'archived' ? (
            <button onClick={() => handleStatus('inbox')} disabled={updating}
              className="px-6 py-2.5 rounded-full bg-surface-container-high text-on-surface font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-50 flex items-center gap-2">
              <Icon name="unarchive" size={16} /> Désarchiver
            </button>
          ) : (
            <button onClick={() => handleStatus('archived')} disabled={updating}
              className="px-6 py-2.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-50 flex items-center gap-2">
              <Icon name="archive" size={16} /> Archiver
            </button>
          )}

          {candidate.status === 'interview_scheduled' ? (
            <button onClick={() => setShowInterview(true)}
              className="px-8 py-2.5 rounded-full bg-tertiary/10 text-tertiary font-bold text-sm flex items-center gap-2 hover:bg-tertiary/20 transition-colors">
              <Icon name="event_available" size={16} /> Entretien planifié — Modifier
            </button>
          ) : candidate.decision === 'invite' ? (
            <button onClick={() => setShowInterview(true)} disabled={updating}
              className="px-8 py-2.5 rounded-full bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-lg shadow-primary/20 active:scale-95 transition-transform disabled:opacity-50 flex items-center gap-2">
              <Icon name="event" size={16} /> Planifier l'entretien
            </button>
          ) : (
            <div className={`px-8 py-2.5 rounded-full font-bold text-sm ${colors.bg} ${colors.text}`}>
              {decisionLabel}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left */}
        <div className="col-span-4 space-y-6">
          {/* Profile */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 text-center shadow-ambient">
            <div className={`w-32 h-32 rounded-full mx-auto mb-6 ${AVATAR_COLORS[0]} flex items-center justify-center text-4xl font-black ring-4 ring-surface-container-low`}>
              {ini}
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-on-surface">{candidate.name ?? 'Anonyme'}</h2>
            <p className="text-primary font-medium">{candidate.target_role ?? candidate.sector ?? '—'}</p>
            <p className={`text-sm font-bold mt-1 mb-6 ${decisionColor}`}>{decisionLabel}</p>

            <div className="flex justify-center gap-2 mb-6 flex-wrap">
              {candidate.sector && <span className="px-3 py-1 bg-surface-container rounded-full text-xs font-bold text-on-surface-variant">{candidate.sector}</span>}
              {candidate.years_experience != null && <span className="px-3 py-1 bg-surface-container rounded-full text-xs font-bold text-on-surface-variant">{candidate.years_experience} ans exp.</span>}
            </div>

            <div className="space-y-1 text-left">
              {candidate.email && (
                <a href={`mailto:${candidate.email}?subject=${encodeURIComponent(`Candidature${candidate.target_role ? ' — ' + candidate.target_role : ''} | Luminary ATS`)}&body=${encodeURIComponent(`Bonjour ${candidate.name ?? ''},\n\nNous avons étudié votre candidature avec attention.\n\nCordialement,\nL'équipe RH`)}`}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-surface-container-low transition-colors group">
                  <span className="text-primary"><Icon name="mail" size={18} /></span>
                  <span className="text-sm text-on-surface truncate flex-1">{candidate.email}</span>
                  <span className="text-[10px] font-bold text-primary opacity-0 group-hover:opacity-100 transition-opacity">Contacter →</span>
                </a>
              )}
              {candidate.phone && (
                <a href={`tel:${candidate.phone}`}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-surface-container-low transition-colors group">
                  <span className="text-primary"><Icon name="phone" size={18} /></span>
                  <span className="text-sm text-on-surface flex-1">{candidate.phone}</span>
                  <span className="text-[10px] font-bold text-primary opacity-0 group-hover:opacity-100 transition-opacity">Appeler →</span>
                </a>
              )}
              <div className="flex items-center gap-3 p-3 rounded-lg">
                <span className="text-primary"><Icon name="calendar_today" size={18} /></span>
                <span className="text-sm text-on-surface">Reçu le {date}</span>
              </div>
            </div>
          </section>

          {/* Score */}
          <section className="bg-surface-container-low rounded-2xl p-6 shadow-ambient">
            <h3 className="text-sm font-black text-on-surface uppercase tracking-widest mb-4">Score ML</h3>
            <div className="flex items-center gap-4">
              <div className="relative w-20 h-20 flex-shrink-0">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="34" fill="none" stroke={`${colors.bar}33`} strokeWidth="6" />
                  <circle cx="40" cy="40" r="34" fill="none" stroke={colors.bar} strokeWidth="6"
                    strokeDasharray={`${2 * Math.PI * 34}`}
                    strokeDashoffset={`${2 * Math.PI * 34 * (1 - pct / 100)}`}
                    strokeLinecap="round" />
                </svg>
                <span className={`absolute inset-0 flex items-center justify-center text-lg font-black ${colors.text}`}>{pct}%</span>
              </div>
              <div>
                <p className={`text-xl font-black ${colors.text}`}>{colors.label}</p>
                <p className="text-xs text-on-surface-variant mt-1">Seuil : {Math.round((candidate.threshold_used ?? 0.5) * 100)}%</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right */}
        <div className="col-span-8 space-y-8">
          {/* AI Insights */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 ai-glow relative overflow-hidden shadow-ambient">
            <div className="absolute top-0 right-0 p-6">
              <div className={`flex items-center gap-2 ${colors.bg} px-4 py-2 rounded-full`}>
                <span className={`w-2 h-2 rounded-full animate-pulse`} style={{ backgroundColor: colors.bar }} />
                <span className={`${colors.text} font-black text-xs uppercase tracking-tighter`}>{pct}% Match</span>
              </div>
            </div>
            <div className="flex items-center gap-3 mb-6">
              <span className="text-tertiary"><Icon name="auto_awesome" fill size={30} /></span>
              <h3 className="text-xl font-extrabold text-on-surface">Analyse IA</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-5 bg-surface-container-low rounded-2xl">
                <p className="text-xs font-bold text-tertiary uppercase mb-2">Décision ML</p>
                <p className={`text-lg font-bold ${decisionColor}`}>{decisionLabel}</p>
                <p className="text-xs text-on-surface-variant mt-1">Score {pct}% vs seuil {Math.round((candidate.threshold_used ?? 0.5) * 100)}%</p>
              </div>
              <div className="p-5 bg-surface-container-low rounded-2xl">
                <p className="text-xs font-bold text-tertiary uppercase mb-2">Expérience</p>
                <p className="text-lg font-bold text-on-surface">{candidate.years_experience != null ? `${candidate.years_experience} ans` : '—'}</p>
                <p className="text-xs text-on-surface-variant mt-1">{candidate.sector ?? '—'}</p>
              </div>
              {candidate.eliminated_reason && (
                <div className="p-5 bg-error-container rounded-2xl col-span-2">
                  <p className="text-xs font-bold text-error uppercase mb-2">Raison d'élimination</p>
                  <p className="text-sm text-on-surface font-medium">{candidate.eliminated_reason}</p>
                </div>
              )}
              {shapEntries.length > 0 && (
                <div className="p-5 bg-surface-container-low rounded-2xl col-span-2">
                  <p className="text-xs font-bold text-tertiary uppercase mb-4">Facteurs SHAP (impact sur le score)</p>
                  <div className="space-y-3">
                    {shapEntries.map(([k, v]) => <ShapBar key={k} label={k} value={v} />)}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Infos détaillées */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
            <h3 className="text-xl font-extrabold text-on-surface mb-6">Informations du candidat</h3>
            <div className="grid grid-cols-2 gap-3">
              {infoFields.map(({ label, value }) => (
                <div key={label} className="p-4 bg-surface-container-low rounded-xl">
                  <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mb-1">{label}</p>
                  <p className="text-sm font-semibold text-on-surface">{String(value)}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
