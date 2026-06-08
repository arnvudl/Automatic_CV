import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getJobs, createJob, updateJob, deleteJob, getJobCandidates, getTargetRoles } from '../lib/api'
import { Toast } from '../components/Toast'

const STAGE_LABEL = {
  sourcing:        'Sourcing',
  review:          'Revue CV',
  interview:       'Entretiens',
  final_interview: 'Entretien final',
  closed:          'Clôturé',
}

const PRIORITY_META = {
  high:      { label: 'Priorité haute', bg: 'bg-success/10 text-success' },
  strategic: { label: 'Stratégique',    bg: 'bg-foreground/10 text-foreground' },
  normal:    { label: 'Standard',       bg: 'bg-muted text-muted-foreground' },
}

const STATUS_META = {
  active: { label: 'Actif',     color: 'text-success' },
  draft:  { label: 'Brouillon', color: 'text-muted-foreground' },
  closed: { label: 'Clôturé',  color: 'text-destructive' },
  paused: { label: 'Pause',     color: 'text-warning' },
}

function ScoreColor(score) {
  if (score >= 75) return 'text-success'
  if (score >= 50) return 'text-warning'
  return 'text-muted-foreground'
}

const EMPTY_FORM = {
  title: '', department: '', location: '', description: '',
  status: 'active', stage: 'sourcing', priority: 'normal',
  applicants_count: 0, avg_score: '',
}

function JobModal({ job, onSave, onClose }) {
  const [form, setForm] = useState(job ? {
    title:            job.title           ?? '',
    department:       job.department      ?? '',
    location:         job.location        ?? '',
    description:      job.description     ?? '',
    status:           job.status          ?? 'active',
    stage:            job.stage           ?? 'sourcing',
    priority:         job.priority        ?? 'normal',
    applicants_count: job.applicants_count ?? 0,
    avg_score:        job.avg_score != null ? String(job.avg_score) : '',
  } : { ...EMPTY_FORM })
  const [saving, setSaving] = useState(false)
  const [targetRoles, setTargetRoles] = useState([])

  // Suggestions d'autocomplétion pour le titre du poste, basées sur les
  // rôles cibles déjà détectés dans les CVs reçus (régex côté API).
  useEffect(() => {
    getTargetRoles().then(setTargetRoles).catch(() => setTargetRoles([]))
  }, [])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.title.trim()) return
    setSaving(true)
    try {
      const payload = {
        ...form,
        applicants_count: Number(form.applicants_count) || 0,
        avg_score: form.avg_score !== '' ? Number(form.avg_score) : null,
      }
      await onSave(payload)
    } finally {
      setSaving(false)
    }
  }

  const inputCls = "w-full bg-muted border border-border rounded-lg px-4 py-2.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 transition-all"
  const labelCls = "block text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2"

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg w-full max-w-lg max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Icon name="work_outline" size={20} className="text-foreground" />
            <h3 className="text-lg font-bold text-foreground">
              {job ? "Modifier l'offre" : 'Créer une offre'}
            </h3>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-muted rounded-lg text-muted-foreground transition-colors">
            <Icon name="close" size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className={labelCls}>Titre du poste *</label>
            <input value={form.title} onChange={e => set('title', e.target.value)}
              placeholder="ex. Développeur Full-Stack Senior"
              list="job-title-suggestions"
              className={inputCls} required />
            <datalist id="job-title-suggestions">
              {targetRoles.map(r => <option key={r} value={r} />)}
            </datalist>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>Département</label>
              <input value={form.department} onChange={e => set('department', e.target.value)}
                placeholder="ex. Ingénierie" className={inputCls} />
            </div>
            <div>
              <label className={labelCls}>Localisation</label>
              <input value={form.location} onChange={e => set('location', e.target.value)}
                placeholder="ex. Paris / Remote" className={inputCls} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>Priorité</label>
              <select value={form.priority} onChange={e => set('priority', e.target.value)} className={inputCls}>
                <option value="normal">Standard</option>
                <option value="high">Haute priorité</option>
                <option value="strategic">Stratégique</option>
              </select>
            </div>
            <div>
              <label className={labelCls}>Statut</label>
              <select value={form.status} onChange={e => set('status', e.target.value)} className={inputCls}>
                <option value="active">Actif</option>
                <option value="draft">Brouillon</option>
                <option value="paused">En pause</option>
                <option value="closed">Clôturé</option>
              </select>
            </div>
          </div>

          <div>
            <label className={labelCls}>Étape du recrutement</label>
            <select value={form.stage} onChange={e => set('stage', e.target.value)} className={inputCls}>
              <option value="sourcing">Sourcing</option>
              <option value="review">Revue CV</option>
              <option value="interview">Entretiens</option>
              <option value="final_interview">Entretien final</option>
              <option value="closed">Clôturé</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>Nb candidats</label>
              <input type="number" min="0" value={form.applicants_count}
                onChange={e => set('applicants_count', e.target.value)} className={inputCls} />
            </div>
            <div>
              <label className={labelCls}>Score IA moyen (%)</label>
              <input type="number" min="0" max="100" step="1" value={form.avg_score}
                onChange={e => set('avg_score', e.target.value)}
                placeholder="—" className={inputCls} />
            </div>
          </div>

          <div>
            <label className={labelCls}>Description (optionnel)</label>
            <textarea value={form.description} onChange={e => set('description', e.target.value)}
              placeholder="Contexte, missions principales, profil recherché..."
              rows={3} className={`${inputCls} resize-none`} />
          </div>

          <div className="flex gap-3 justify-end pt-2">
            <button type="button" onClick={onClose}
              className="px-5 py-2.5 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors">
              Annuler
            </button>
            <button type="submit" disabled={saving || !form.title.trim()}
              className="px-6 py-2.5 rounded-lg bg-foreground text-primary-foreground font-bold text-sm hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2">
              <Icon name={job ? 'save' : 'add'} size={16} />
              {saving ? 'Enregistrement...' : (job ? 'Enregistrer' : "Créer l'offre")}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Jobs({ onNavigate }) {
  const [jobs, setJobs]           = useState([])
  const [loading, setLoading]     = useState(true)
  const [modal, setModal]         = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [toast, setToast]         = useState(null)

  // Liste des candidats par offre (lazy-loaded à l'expansion)
  const [expandedJobId, setExpandedJobId]   = useState(null)
  const [jobCandidates, setJobCandidates]   = useState({})   // { job_id: [...] | 'loading' | 'error' }

  const showToast = (message, type = 'success') => setToast({ message, type })

  const toggleCandidates = async (jobId) => {
    if (expandedJobId === jobId) {
      setExpandedJobId(null)
      return
    }
    setExpandedJobId(jobId)
    if (!jobCandidates[jobId]) {
      setJobCandidates(prev => ({ ...prev, [jobId]: 'loading' }))
      try {
        const data = await getJobCandidates(jobId)
        setJobCandidates(prev => ({ ...prev, [jobId]: data }))
      } catch {
        setJobCandidates(prev => ({ ...prev, [jobId]: 'error' }))
      }
    }
  }

  useEffect(() => {
    getJobs()
      .then(jobsData => {
        setJobs(jobsData)
        // Précharge la liste des candidats de chaque offre — le compteur
        // "Candidats" affiché sur la carte doit refléter le total réel
        // du pipeline (assignés + auto-détectés en Inbox), pas un
        // compteur statique potentiellement désynchronisé.
        jobsData.forEach(j => {
          getJobCandidates(j.job_id)
            .then(data => setJobCandidates(prev => ({ ...prev, [j.job_id]: data })))
            .catch(() => {})
        })
      })
      .catch(() => setJobs([]))
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async (payload) => {
    try {
      if (modal?.mode === 'edit') {
        const updated = await updateJob(modal.job.job_id, payload)
        setJobs(prev => prev.map(j => j.job_id === updated.job_id ? updated : j))
        showToast('Offre mise à jour')
      } else {
        const created = await createJob(payload)
        setJobs(prev => [created, ...prev])
        showToast('Offre créée avec succès')
      }
      setModal(null)
    } catch {
      showToast("Erreur lors de l'enregistrement", 'error')
    }
  }

  const handleDelete = async (id) => {
    try {
      await deleteJob(id)
      setJobs(prev => prev.filter(j => j.job_id !== id))
      showToast('Offre supprimée', 'info')
    } catch {
      showToast('Erreur lors de la suppression', 'error')
    } finally {
      setConfirmDelete(null)
    }
  }

  const totalApplicants = jobs.reduce((s, j) => {
    const real = jobCandidates[j.job_id]
    return s + (Array.isArray(real) ? real.length : (j.applicants_count || 0))
  }, 0)
  const activeJobs      = jobs.filter(j => j.status === 'active').length
  const withScore       = jobs.filter(j => j.avg_score != null)
  const avgScore        = withScore.length > 0
    ? Math.round(withScore.reduce((s, j) => s + j.avg_score, 0) / withScore.length)
    : null

  return (
    <div className="p-10 min-h-screen">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {/* Confirm delete */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={() => setConfirmDelete(null)}>
          <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-destructive"><Icon name="warning" size={28} /></span>
              <h3 className="text-lg font-bold text-foreground">Supprimer l'offre ?</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-6">Cette action est irréversible.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)}
                className="px-5 py-2 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors">
                Annuler
              </button>
              <button onClick={() => handleDelete(confirmDelete)}
                className="px-5 py-2 rounded-lg bg-destructive text-white font-semibold text-sm hover:opacity-90 transition-opacity flex items-center gap-2">
                <Icon name="delete_forever" size={16} /> Supprimer
              </button>
            </div>
          </div>
        </div>
      )}

      {modal && (
        <JobModal
          job={modal.mode === 'edit' ? modal.job : null}
          onSave={handleSave}
          onClose={() => setModal(null)}
        />
      )}

      {/* Header */}
      <div className="flex justify-between items-start mb-10">
        <div>
          <h1 className="text-3xl font-bold text-foreground tracking-tight mb-1">Offres d'emploi</h1>
          <p className="text-muted-foreground text-sm max-w-2xl">
            Gérez vos cycles de recrutement et identifiez les meilleurs talents.
          </p>
        </div>
        <button onClick={() => setModal({ mode: 'create' })}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-foreground text-primary-foreground font-bold text-sm hover:opacity-90 transition-opacity">
          <Icon name="add" size={18} />
          Nouvelle offre
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
        <div className="bg-card border border-border p-6 rounded-xl shadow-card">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3">Candidatures totales</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-bold text-foreground">{loading ? '—' : totalApplicants.toLocaleString('fr-FR')}</span>
          </div>
          <p className="text-muted-foreground text-xs mt-2">{activeJobs} offre{activeJobs !== 1 ? 's' : ''} active{activeJobs !== 1 ? 's' : ''}</p>
        </div>

        <div className="bg-card border border-border p-6 rounded-xl shadow-card">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3">Score IA moyen</p>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full border-4 border-success/20 border-t-success flex items-center justify-center">
              <span className="text-success font-bold text-sm">{avgScore ?? '—'}</span>
            </div>
            {avgScore && <span className="text-foreground text-2xl font-bold">%</span>}
          </div>
          <p className="text-muted-foreground text-xs mt-2">
            {avgScore != null ? (avgScore >= 75 ? 'Excellente qualité' : avgScore >= 50 ? 'Bonne qualité' : 'À améliorer') : 'Aucune donnée'}
          </p>
        </div>

        <div className="bg-foreground p-6 rounded-xl shadow-card text-primary-foreground">
          <p className="text-xs font-semibold opacity-60 uppercase tracking-widest mb-3">Offres actives</p>
          <div className="text-4xl font-bold">{loading ? '—' : activeJobs}</div>
          <p className="text-xs opacity-60 mt-2">sur {jobs.length} au total</p>
        </div>
      </div>

      {/* Jobs Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-24 text-muted-foreground gap-3">
          <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
          <span className="font-medium">Chargement...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {jobs.map((job) => {
            const pm   = PRIORITY_META[job.priority] ?? PRIORITY_META.normal
            const sm   = STATUS_META[job.status]     ?? STATUS_META.active
            const isDraft = job.status === 'draft'
            return (
              <div key={job.job_id}
                className={`bg-card border border-border rounded-xl p-6 shadow-card hover:shadow-card-md transition-all group flex flex-col h-full ${isDraft ? 'opacity-70' : ''}`}>
                <div className="flex justify-between items-start mb-5">
                  <span className={`${pm.bg} text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-wider`}>
                    {pm.label}
                  </span>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button onClick={() => setModal({ mode: 'edit', job })}
                      className="p-1.5 hover:bg-muted rounded-lg text-muted-foreground transition-colors" title="Modifier">
                      <Icon name="edit" size={15} />
                    </button>
                    <button onClick={() => setConfirmDelete(job.job_id)}
                      className="p-1.5 hover:bg-destructive/10 rounded-lg text-muted-foreground hover:text-destructive transition-colors" title="Supprimer">
                      <Icon name="delete" size={15} />
                    </button>
                  </div>
                </div>

                <h3 className="text-base font-bold text-foreground leading-snug mb-1">
                  {job.title}
                </h3>
                <p className="text-muted-foreground text-sm mb-0.5">
                  {[job.department, job.location].filter(Boolean).join(' · ') || '—'}
                </p>
                <p className={`text-xs font-semibold mb-5 ${sm.color}`}>{sm.label}</p>

                <div className={`grid grid-cols-2 gap-3 mb-3 ${isDraft ? 'grayscale' : ''}`}>
                  <button
                    onClick={() => toggleCandidates(job.job_id)}
                    className="bg-muted rounded-lg p-4 text-left hover:bg-muted/70 transition-colors group/cand">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Candidats</p>
                      <Icon name={expandedJobId === job.job_id ? 'expand_less' : 'expand_more'} size={14}
                        className="text-muted-foreground group-hover/cand:text-foreground transition-colors" />
                    </div>
                    <p className="text-2xl font-bold text-foreground">
                      {Array.isArray(jobCandidates[job.job_id])
                        ? jobCandidates[job.job_id].length
                        : (job.applicants_count ?? 0)}
                    </p>
                  </button>
                  <div className="bg-muted rounded-lg p-4">
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Score IA</p>
                    <p className={`text-2xl font-bold ${job.avg_score != null ? ScoreColor(job.avg_score) : 'text-muted-foreground'}`}>
                      {job.avg_score != null ? `${Math.round(job.avg_score)}%` : '—'}
                    </p>
                  </div>
                </div>

                {expandedJobId === job.job_id && (
                  <div className="mb-3 bg-muted/50 border border-border rounded-lg p-3 max-h-56 overflow-y-auto">
                    {jobCandidates[job.job_id] === 'loading' && (
                      <div className="flex items-center gap-2 text-xs text-muted-foreground py-2 px-1">
                        <div className="w-3.5 h-3.5 border-2 border-border border-t-foreground rounded-full animate-spin" />
                        Chargement…
                      </div>
                    )}
                    {jobCandidates[job.job_id] === 'error' && (
                      <p className="text-xs text-destructive px-1 py-2">Erreur de chargement.</p>
                    )}
                    {Array.isArray(jobCandidates[job.job_id]) && (
                      jobCandidates[job.job_id].length === 0 ? (
                        <p className="text-xs text-muted-foreground px-1 py-2">Aucun candidat pour cette offre pour l'instant.</p>
                      ) : (
                        <ul className="space-y-1">
                          {jobCandidates[job.job_id].map(c => (
                            <li key={c.candidate_id}>
                              <button
                                onClick={() => onNavigate?.('profile', c.candidate_id)}
                                className="w-full flex items-center justify-between gap-2 px-2 py-1.5 rounded-md hover:bg-card text-left transition-colors group/c">
                                <div className="min-w-0">
                                  <div className="flex items-center gap-1.5">
                                    <p className="text-sm font-semibold text-foreground truncate group-hover/c:underline">{c.name || 'Candidat sans nom'}</p>
                                    {c.unassigned && (
                                      <span className="flex-shrink-0 text-[8px] font-bold uppercase tracking-wider px-1 py-0.5 rounded bg-foreground/10 text-foreground"
                                        title="Candidat scoré, pas encore assigné à cette offre">
                                        Nouveau
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-[11px] text-muted-foreground truncate">{c.stage_name || '—'}{c.target_role ? ` · ${c.target_role}` : ''}</p>
                                </div>
                                {c.score != null && (
                                  <span className={`flex-shrink-0 text-xs font-bold px-2 py-0.5 rounded-full ${
                                    c.decision === 'invite' ? 'bg-success/10 text-success'
                                    : c.decision === 'eliminated' ? 'bg-muted text-muted-foreground'
                                    : 'bg-warning/10 text-warning'
                                  }`}>
                                    {Math.round(c.score * 100)}%
                                  </span>
                                )}
                              </button>
                            </li>
                          ))}
                        </ul>
                      )
                    )}
                  </div>
                )}

                <div className="mt-auto">
                  <div className="flex justify-between items-end mb-1.5">
                    <p className="text-xs text-muted-foreground">Étape</p>
                    <p className={`text-xs font-semibold ${isDraft ? 'text-muted-foreground/40 italic' : 'text-foreground'}`}>
                      {STAGE_LABEL[job.stage] ?? job.stage}
                    </p>
                  </div>
                  <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-foreground rounded-full transition-all"
                      style={{ width: `${job.progress ?? 0}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          })}

          {/* Create card */}
          <button onClick={() => setModal({ mode: 'create' })}
            className="border-2 border-dashed border-border rounded-xl p-6 hover:border-foreground/30 hover:bg-muted/50 transition-all flex flex-col items-center justify-center group min-h-[280px]">
            <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mb-3 group-hover:bg-foreground group-hover:text-primary-foreground transition-all text-muted-foreground">
              <Icon name="add" size={28} />
            </div>
            <span className="text-base font-bold text-foreground">Créer une offre</span>
            <span className="text-sm text-muted-foreground mt-1 text-center">Démarrer un nouveau cycle de recrutement</span>
          </button>
        </div>
      )}
    </div>
  )
}
