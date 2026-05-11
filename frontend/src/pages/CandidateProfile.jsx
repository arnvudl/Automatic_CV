import { useState, useEffect, useCallback } from 'react'
import { Icon } from '../components/Icon'
import { getCandidate, updateStatus, createInterview, getExplain } from '../lib/api'
import { Toast } from '../components/Toast'
import { InterviewModal } from '../components/InterviewModal'
import { ScorecardPanel } from '../components/ScorecardPanel'

const AVATAR_COLORS = [
  'bg-blue-100 text-blue-700',
  'bg-violet-100 text-violet-700',
  'bg-emerald-100 text-emerald-700',
  'bg-purple-100 text-purple-700',
  'bg-amber-100 text-amber-700',
]

function initials(name) {
  return (name ?? '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase() || '??'
}

function getScoreColors(score) {
  const pct = Math.round((score ?? 0) * 100)
  if (pct >= 75) return { text: 'text-success',     bg: 'bg-success/10',     bar: '#16a34a', label: 'Top Match' }
  if (pct >= 50) return { text: 'text-warning',     bg: 'bg-warning/10',     bar: '#ca8a04', label: 'Good Match' }
  return              { text: 'text-destructive', bg: 'bg-destructive/10', bar: '#dc2626', label: 'Low Match' }
}

function ShapBar({ label, value }) {
  const abs      = Math.abs(value)
  const width    = Math.min((abs / 0.3) * 100, 100)
  const positive = value >= 0
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-36 text-muted-foreground font-medium truncate">{label}</span>
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${positive ? 'bg-success' : 'bg-destructive'}`} style={{ width: `${width}%` }} />
      </div>
      <span className={`w-12 text-right font-bold ${positive ? 'text-success' : 'text-destructive'}`}>
        {positive ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  )
}

// ── Chip de compétence ────────────────────────────────────────────────
function SkillChip({ label, variant = 'default' }) {
  const styles = {
    default: 'bg-muted text-muted-foreground',
    tech:    'bg-blue-50 text-blue-700 border border-blue-100',
    meth:    'bg-violet-50 text-violet-700 border border-violet-100',
    mgmt:    'bg-amber-50 text-amber-700 border border-amber-100',
    lang:    'bg-emerald-50 text-emerald-700 border border-emerald-100',
  }
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-semibold ${styles[variant]}`}>
      {label}
    </span>
  )
}

export default function CandidateProfile({ onNavigate, candidateId }) {
  const [candidate, setCandidate]       = useState(null)
  const [loading, setLoading]           = useState(true)
  const [error, setError]               = useState(null)
  const [updating, setUpdating]         = useState(false)
  const [toast, setToast]               = useState(null)
  const [showInterview, setShowInterview] = useState(false)
  const [explain, setExplain]           = useState(null)
  const [loadingExplain, setLoadingExplain] = useState(false)
  const [showExplain, setShowExplain]   = useState(false)

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

  const handleExplain = async () => {
    if (explain) { setShowExplain(v => !v); return }
    setLoadingExplain(true)
    setShowExplain(true)
    try {
      const data = await getExplain(candidateId)
      setExplain(data)
    } catch {
      showToast('Explication SHAP indisponible', 'error')
      setShowExplain(false)
    } finally {
      setLoadingExplain(false)
    }
  }

  const handleStatus = async (newStatus) => {
    if (!candidate || updating) return
    setUpdating(true)
    try {
      await updateStatus(candidate.candidate_id, newStatus)
      setCandidate(prev => ({ ...prev, status: newStatus }))
      const messages = {
        archived:            'Candidat archivé',
        interview_scheduled: 'Entretien planifié ✓',
        review:              'Mis en révision',
        rejected:            'Candidat rejeté',
        inbox:               'Remis en boîte de réception',
      }
      showToast(messages[newStatus] ?? 'Statut mis à jour')
    } catch {
      showToast('Erreur lors de la mise à jour', 'error')
    } finally {
      setUpdating(false)
    }
  }

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh] text-muted-foreground gap-3">
      <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
      <span className="font-medium">Chargement...</span>
    </div>
  )

  if (error || !candidate) return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-muted-foreground">
      <Icon name="person_off" size={48} />
      <p className="font-bold text-foreground">{error ?? 'Candidat introuvable'}</p>
      <button onClick={() => onNavigate('candidates')} className="text-foreground font-bold text-sm hover:opacity-70 transition-opacity">
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
  const decisionColor = {
    invite:     'text-success',
    reject:     'text-destructive',
    eliminated: 'text-muted-foreground',
  }[candidate.decision] ?? 'text-muted-foreground'

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
          onConfirm={async ({ date: d, time, type, notes }) => {
            setShowInterview(false)
            await handleStatus('interview_scheduled')
            try {
              await createInterview({
                candidate_id:   candidate.candidate_id,
                candidate_name: candidate.name,
                date: d, time, type, notes,
              })
            } catch (_) {}
            showToast(`Entretien ${type} planifié le ${new Date(d).toLocaleDateString('fr-FR')} à ${time}`)
          }}
        />
      )}

      {/* Breadcrumb + Actions */}
      <div className="flex justify-between items-center mb-10">
        <button onClick={() => onNavigate('candidates')}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors text-sm font-medium">
          <Icon name="arrow_back" size={18} />
          Retour aux candidats
        </button>
        <div className="flex gap-3 flex-wrap">
          {candidate.status === 'archived' ? (
            <button onClick={() => handleStatus('inbox')} disabled={updating}
              className="px-5 py-2 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors disabled:opacity-50 flex items-center gap-2">
              <Icon name="unarchive" size={16} /> Désarchiver
            </button>
          ) : (
            <button onClick={() => handleStatus('archived')} disabled={updating}
              className="px-5 py-2 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors disabled:opacity-50 flex items-center gap-2">
              <Icon name="archive" size={16} /> Archiver
            </button>
          )}

          {candidate.status === 'interview_scheduled' ? (
            <button onClick={() => setShowInterview(true)}
              className="px-6 py-2 rounded-lg bg-success/10 text-success font-bold text-sm flex items-center gap-2 hover:bg-success/20 transition-colors">
              <Icon name="event_available" size={16} /> Entretien planifié — Modifier
            </button>
          ) : candidate.decision === 'invite' ? (
            <button onClick={() => setShowInterview(true)} disabled={updating}
              className="px-6 py-2 rounded-lg bg-foreground text-primary-foreground font-bold text-sm disabled:opacity-50 flex items-center gap-2 hover:opacity-90 transition-opacity">
              <Icon name="event" size={16} /> Planifier l'entretien
            </button>
          ) : (
            <div className={`px-6 py-2 rounded-lg font-bold text-sm ${colors.bg} ${colors.text}`}>
              {decisionLabel}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left */}
        <div className="col-span-4 space-y-6">
          {/* Profile card */}
          <section className="bg-card border border-border rounded-xl p-8 text-center shadow-card">
            <div className={`w-24 h-24 rounded-full mx-auto mb-5 ${AVATAR_COLORS[0]} flex items-center justify-center text-3xl font-black ring-4 ring-border`}>
              {ini}
            </div>
            <h2 className="text-xl font-bold text-foreground">{candidate.name ?? 'Anonyme'}</h2>
            <p className="text-foreground/70 font-medium text-sm mt-0.5">{candidate.target_role ?? candidate.sector ?? '—'}</p>
            <p className={`text-sm font-bold mt-1 mb-5 ${decisionColor}`}>{decisionLabel}</p>

            <div className="flex justify-center gap-2 mb-5 flex-wrap">
              {candidate.sector && <span className="px-3 py-1 bg-muted rounded-full text-xs font-semibold text-muted-foreground">{candidate.sector}</span>}
              {candidate.years_experience != null && <span className="px-3 py-1 bg-muted rounded-full text-xs font-semibold text-muted-foreground">{candidate.years_experience} ans exp.</span>}
            </div>

            <div className="space-y-1 text-left">
              {candidate.email && (
                <a href={`mailto:${candidate.email}?subject=${encodeURIComponent(`Candidature${candidate.target_role ? ' — ' + candidate.target_role : ''} | Luminary ATS`)}&body=${encodeURIComponent(`Bonjour ${candidate.name ?? ''},\n\nNous avons étudié votre candidature avec attention.\n\nCordialement,\nL'équipe RH`)}`}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors group">
                  <span className="text-foreground"><Icon name="mail" size={16} /></span>
                  <span className="text-sm text-foreground truncate flex-1">{candidate.email}</span>
                  <span className="text-[10px] font-bold text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">Contacter →</span>
                </a>
              )}
              {candidate.phone && (
                <a href={`tel:${candidate.phone}`}
                  className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors group">
                  <span className="text-foreground"><Icon name="phone" size={16} /></span>
                  <span className="text-sm text-foreground flex-1">{candidate.phone}</span>
                  <span className="text-[10px] font-bold text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">Appeler →</span>
                </a>
              )}
              <div className="flex items-center gap-3 p-3 rounded-lg">
                <span className="text-foreground"><Icon name="calendar_today" size={16} /></span>
                <span className="text-sm text-foreground">Reçu le {date}</span>
              </div>
            </div>
          </section>

          {/* Score card */}
          <section className="bg-card border border-border rounded-xl p-6 shadow-card">
            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">Score ML</h3>
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
                <p className="text-xs text-muted-foreground mt-1">Seuil : {Math.round((candidate.threshold_used ?? 0.5) * 100)}%</p>
              </div>
            </div>
          </section>
        </div>

        {/* Right */}
        <div className="col-span-8 space-y-6">

          {/* Score ML + Décision */}
          <section className="bg-card border border-border rounded-xl p-8 shadow-card">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Icon name="auto_awesome" fill size={22} className="text-foreground" />
                <h3 className="text-lg font-bold text-foreground">Score ML</h3>
              </div>
              <div className={`flex items-center gap-2 ${colors.bg} px-3 py-1.5 rounded-full`}>
                <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: colors.bar }} />
                <span className={`${colors.text} font-bold text-xs uppercase`}>{pct}% Match</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-4 bg-muted rounded-xl">
                <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Décision</p>
                <p className={`text-base font-bold ${decisionColor}`}>{decisionLabel}</p>
                <p className="text-xs text-muted-foreground mt-1">Score {pct}% vs seuil {Math.round((candidate.threshold_used ?? 0.5) * 100)}%</p>
              </div>
              <div className="p-4 bg-muted rounded-xl">
                <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Expérience</p>
                <p className="text-base font-bold text-foreground">{candidate.years_experience != null ? `${candidate.years_experience} ans` : '—'}</p>
                <p className="text-xs text-muted-foreground mt-1">{candidate.sector ?? '—'}</p>
              </div>
              {candidate.eliminated_reason && (
                <div className="p-4 bg-destructive/10 rounded-xl col-span-2">
                  <p className="text-[10px] font-bold text-destructive uppercase tracking-widest mb-2">Raison d'élimination</p>
                  <p className="text-sm text-foreground font-medium">{candidate.eliminated_reason}</p>
                </div>
              )}
            </div>

            {/* Bouton explication SHAP */}
            <button
              onClick={handleExplain}
              className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground transition-colors"
            >
              <Icon name={showExplain ? 'expand_less' : 'expand_more'} size={18} />
              {showExplain ? 'Masquer' : 'Voir'} l'explication détaillée des facteurs
            </button>

            {showExplain && (
              <div className="mt-4 border-t border-border pt-4">
                {loadingExplain ? (
                  <div className="flex items-center gap-2 text-muted-foreground py-2">
                    <div className="w-4 h-4 border-2 border-border border-t-foreground rounded-full animate-spin" />
                    <span className="text-sm">Calcul en cours…</span>
                  </div>
                ) : explain ? (
                  <div className="space-y-4">
                    {/* Narrative */}
                    {explain.narrative && (
                      <p className="text-sm text-muted-foreground leading-relaxed bg-muted rounded-xl px-4 py-3">
                        {explain.narrative}
                      </p>
                    )}
                    {/* Barres SHAP */}
                    {explain.shap && Object.keys(explain.shap).length > 0 ? (
                      <div>
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Impact des facteurs sur le score</p>
                        <div className="space-y-3">
                          {Object.entries(explain.shap)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 8)
                            .map(([k, v]) => <ShapBar key={k} label={k} value={v} />)}
                        </div>
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground italic">
                        Données SHAP non disponibles — elles apparaissent uniquement pour les CVs scorés via le pipeline ML.
                      </p>
                    )}
                    {/* Facteurs manquants */}
                    {explain.missing?.length > 0 && (
                      <div>
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Points à améliorer vs candidats invités</p>
                        <div className="space-y-2">
                          {explain.missing.map(m => (
                            <div key={m.feature} className="flex items-center justify-between bg-destructive/5 rounded-lg px-3 py-2">
                              <span className="text-xs font-semibold text-foreground">{m.feature}</span>
                              <span className="text-xs text-destructive font-bold">−{m.gap_pct}% vs moyenne</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            )}
          </section>

          {/* Compétences & Profil CV */}
          {(candidate.summary || candidate.skills_tech?.length > 0 || candidate.skills_meth?.length > 0 ||
            candidate.skills_mgmt?.length > 0 || candidate.languages?.length > 0 || candidate.certifications?.length > 0) && (
            <section className="bg-card border border-border rounded-xl p-8 shadow-card">
              <h3 className="text-lg font-bold text-foreground mb-5">Compétences & Profil</h3>

              {candidate.summary && (
                <div className="mb-5">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Résumé</p>
                  <p className="text-sm text-foreground leading-relaxed">{candidate.summary}</p>
                </div>
              )}

              {candidate.skills_tech?.length > 0 && (
                <div className="mb-4">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Compétences techniques</p>
                  <div className="flex flex-wrap gap-1.5">
                    {candidate.skills_tech.map(s => <SkillChip key={s} label={s} variant="tech" />)}
                  </div>
                </div>
              )}

              {candidate.skills_meth?.length > 0 && (
                <div className="mb-4">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Méthodes & outils</p>
                  <div className="flex flex-wrap gap-1.5">
                    {candidate.skills_meth.map(s => <SkillChip key={s} label={s} variant="meth" />)}
                  </div>
                </div>
              )}

              {candidate.skills_mgmt?.length > 0 && (
                <div className="mb-4">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Management & soft skills</p>
                  <div className="flex flex-wrap gap-1.5">
                    {candidate.skills_mgmt.map(s => <SkillChip key={s} label={s} variant="mgmt" />)}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                {candidate.languages?.length > 0 && (
                  <div>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Langues</p>
                    <div className="flex flex-wrap gap-1.5">
                      {candidate.languages.map(l => <SkillChip key={l} label={l} variant="lang" />)}
                    </div>
                  </div>
                )}
                {candidate.certifications?.length > 0 && (
                  <div>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Certifications</p>
                    <ul className="space-y-1">
                      {candidate.certifications.map(c => (
                        <li key={c} className="text-xs text-foreground flex items-start gap-1.5">
                          <Icon name="verified" size={13} className="text-success mt-0.5 flex-shrink-0" />
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Informations */}
          <section className="bg-card border border-border rounded-xl p-8 shadow-card">
            <h3 className="text-lg font-bold text-foreground mb-5">Informations de contact</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Email',        value: candidate.email },
                { label: 'Téléphone',    value: candidate.phone },
                { label: 'Secteur',      value: candidate.sector },
                { label: 'Rôle cible',   value: candidate.target_role },
                { label: 'Expérience',   value: candidate.years_experience != null ? `${candidate.years_experience} ans` : null },
                { label: 'Statut',       value: candidate.status },
                { label: 'Reçu le',      value: date },
              ].filter(f => f.value != null).map(({ label, value }) => (
                <div key={label} className="p-4 bg-muted rounded-xl">
                  <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">{label}</p>
                  <p className="text-sm font-semibold text-foreground">{String(value)}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Évaluation RH */}
          <ScorecardPanel candidateId={candidateId} />
        </div>
      </div>
    </div>
  )
}
