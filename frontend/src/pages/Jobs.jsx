import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getJobs, createJob, updateJob, deleteJob } from '../lib/api'
import { Toast } from '../components/Toast'

// ── Helpers ────────────────────────────────────────────────────────
const STAGE_LABEL = {
  sourcing:         'Sourcing',
  review:           'Revue CV',
  interview:        'Entretiens',
  final_interview:  'Entretien final',
  closed:           'Clôturé',
}

const PRIORITY_META = {
  high:      { label: 'Priorité haute',  bg: 'bg-tertiary-container/10 text-tertiary' },
  strategic: { label: 'Stratégique',     bg: 'bg-primary-container/10 text-primary' },
  normal:    { label: 'Standard',        bg: 'bg-secondary-container/20 text-on-secondary-container' },
}

const STATUS_META = {
  active:  { label: 'Actif',   color: 'text-tertiary' },
  draft:   { label: 'Brouillon', color: 'text-on-surface-variant' },
  closed:  { label: 'Clôturé',  color: 'text-error' },
  paused:  { label: 'Pause',    color: 'text-amber-600' },
}

function ScoreColor(score) {
  if (score >= 75) return 'text-tertiary'
  if (score >= 50) return 'text-amber-600'
  return 'text-on-surface-variant'
}

// ── Job Modal ──────────────────────────────────────────────────────
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

  const inputCls = "w-full bg-surface-container-low border-none rounded-xl px-4 py-3 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/30 transition-all"
  const labelCls = "block text-xs font-bold text-on-surface-variant uppercase tracking-widest mb-2"

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-surface-container-lowest rounded-3xl p-8 shadow-ambient-lg w-full max-w-lg max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <span className="text-primary"><Icon name="work_outline" size={24} /></span>
            <h3 className="text-xl font-extrabold text-on-surface">
              {job ? 'Modifier l\'offre' : 'Créer une offre'}
            </h3>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-surface-container-high rounded-xl text-on-surface-variant transition-colors">
            <Icon name="close" size={20} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Titre */}
          <div>
            <label className={labelCls}>Titre du poste *</label>
            <input value={form.title} onChange={e => set('title', e.target.value)}
              placeholder="ex. Développeur Full-Stack Senior"
              className={inputCls} required />
          </div>

          {/* Dept + Localisation */}
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

          {/* Priorité + Statut */}
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

          {/* Étape */}
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

          {/* Candidats + Score */}
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

          {/* Description */}
          <div>
            <label className={labelCls}>Description (optionnel)</label>
            <textarea value={form.description} onChange={e => set('description', e.target.value)}
              placeholder="Contexte, missions principales, profil recherché..."
              rows={3} className={`${inputCls} resize-none`} />
          </div>

          {/* Actions */}
          <div className="flex gap-3 justify-end pt-2">
            <button type="button" onClick={onClose}
              className="px-5 py-2.5 rounded-xl bg-surface-container text-on-surface font-semibold text-sm hover:bg-surface-container-high transition-colors">
              Annuler
            </button>
            <button type="submit" disabled={saving || !form.title.trim()}
              className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-sm hover:shadow-md active:scale-95 transition-all disabled:opacity-50 flex items-center gap-2">
              <Icon name={job ? 'save' : 'add'} size={16} />
              {saving ? 'Enregistrement...' : (job ? 'Enregistrer' : 'Créer l\'offre')}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────
export default function Jobs() {
  const [jobs, setJobs]       = useState([])
  const [loading, setLoading] = useState(true)
  const [modal, setModal]     = useState(null)   // null | { mode: 'create' } | { mode: 'edit', job }
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [toast, setToast]     = useState(null)

  const showToast = (message, type = 'success') => setToast({ message, type })

  useEffect(() => {
    getJobs()
      .then(setJobs)
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
      showToast('Erreur lors de l\'enregistrement', 'error')
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

  // ── Stats ──────────────────────────────────────────────────────
  const totalApplicants = jobs.reduce((s, j) => s + (j.applicants_count || 0), 0)
  const activeJobs      = jobs.filter(j => j.status === 'active').length
  const avgScore        = jobs.filter(j => j.avg_score != null).length > 0
    ? Math.round(jobs.filter(j => j.avg_score != null).reduce((s, j) => s + j.avg_score, 0) / jobs.filter(j => j.avg_score != null).length)
    : null

  return (
    <div className="p-10 min-h-screen">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {/* Confirm delete modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={() => setConfirmDelete(null)}>
          <div className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient-lg max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-error"><Icon name="warning" size={28} /></span>
              <h3 className="text-lg font-bold text-on-surface">Supprimer l'offre ?</h3>
            </div>
            <p className="text-sm text-on-surface-variant mb-6">Cette action est irréversible.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)}
                className="px-5 py-2 rounded-xl bg-surface-container text-on-surface font-semibold text-sm">
                Annuler
              </button>
              <button onClick={() => handleDelete(confirmDelete)}
                className="px-5 py-2 rounded-xl bg-error text-white font-semibold text-sm flex items-center gap-2">
                <Icon name="delete_forever" size={16} /> Supprimer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Job modal */}
      {modal && (
        <JobModal
          job={modal.mode === 'edit' ? modal.job : null}
          onSave={handleSave}
          onClose={() => setModal(null)}
        />
      )}

      {/* Header */}
      <div className="flex justify-between items-start mb-12">
        <div>
          <h1 className="text-6xl font-black text-on-surface tracking-tight leading-tight mb-2">Offres d'emploi</h1>
          <p className="text-on-surface-variant text-lg max-w-2xl font-medium opacity-70">
            Gérez vos cycles de recrutement et identifiez les meilleurs talents.
          </p>
        </div>
        <button onClick={() => setModal({ mode: 'create' })}
          className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-lg shadow-primary/20 hover:shadow-xl active:scale-95 transition-all">
          <Icon name="add" size={20} />
          Nouvelle offre
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
        <div className="md:col-span-2 bg-surface-container-lowest p-8 rounded-3xl shadow-ambient relative overflow-hidden group">
          <div className="absolute -right-12 -top-12 w-48 h-48 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-colors" />
          <p className="text-sm font-bold text-primary tracking-widest uppercase mb-4">Candidatures totales</p>
          <div className="flex items-baseline gap-4">
            <span className="text-6xl font-black text-on-surface">{loading ? '...' : totalApplicants.toLocaleString('fr-FR')}</span>
          </div>
          <p className="text-on-surface-variant text-sm mt-4 font-medium">{activeJobs} offre{activeJobs > 1 ? 's' : ''} active{activeJobs > 1 ? 's' : ''}</p>
        </div>

        <div className="bg-surface-container-lowest p-8 rounded-3xl shadow-ambient">
          <p className="text-sm font-bold text-on-surface-variant/60 tracking-widest uppercase mb-4">Score IA moyen</p>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full border-4 border-tertiary/20 border-t-tertiary flex items-center justify-center">
              <span className="text-tertiary font-bold text-lg">{avgScore ?? '—'}</span>
            </div>
            {avgScore && <span className="text-on-surface text-3xl font-black">%</span>}
          </div>
          <p className="text-on-surface-variant text-sm mt-4 font-medium">
            {avgScore != null ? (avgScore >= 75 ? 'Excellente qualité' : avgScore >= 50 ? 'Bonne qualité' : 'À améliorer') : 'Aucune donnée'}
          </p>
        </div>

        <div className="bg-primary-container p-8 rounded-3xl shadow-lg text-on-primary-container">
          <p className="text-sm font-bold tracking-widest uppercase mb-4 opacity-80">Offres actives</p>
          <div className="text-5xl font-black">{loading ? '...' : activeJobs}</div>
          <p className="text-sm opacity-70 mt-4 font-medium">sur {jobs.length} au total</p>
        </div>
      </div>

      {/* Jobs Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-24 text-on-surface-variant gap-3">
          <Icon name="hourglass_empty" size={32} />
          <span className="font-medium">Chargement...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {jobs.map((job) => {
            const pm   = PRIORITY_META[job.priority] ?? PRIORITY_META.normal
            const sm   = STATUS_META[job.status]     ?? STATUS_META.active
            const isDraft = job.status === 'draft'
            return (
              <div key={job.job_id}
                className={`bg-surface-container-lowest rounded-3xl p-8 shadow-ambient hover:shadow-ambient-lg transition-all group flex flex-col h-full ${isDraft ? 'opacity-70' : ''}`}>
                <div className="flex justify-between items-start mb-6">
                  <span className={`${pm.bg} text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-tighter`}>
                    {pm.label}
                  </span>
                  <div className="flex items-center gap-1">
                    <button onClick={() => setModal({ mode: 'edit', job })}
                      className="p-1.5 hover:bg-surface-container-high rounded-lg text-on-surface-variant opacity-0 group-hover:opacity-100 transition-all"
                      title="Modifier">
                      <Icon name="edit" size={16} />
                    </button>
                    <button onClick={() => setConfirmDelete(job.job_id)}
                      className="p-1.5 hover:bg-error-container rounded-lg text-on-surface-variant hover:text-error opacity-0 group-hover:opacity-100 transition-all"
                      title="Supprimer">
                      <Icon name="delete" size={16} />
                    </button>
                  </div>
                </div>

                <h3 className="text-[1.375rem] font-bold text-on-surface leading-snug group-hover:text-primary transition-colors mb-1">
                  {job.title}
                </h3>
                <p className="text-on-surface-variant text-sm font-medium mb-1">
                  {[job.department, job.location].filter(Boolean).join(' • ') || '—'}
                </p>
                <p className={`text-xs font-bold mb-6 ${sm.color}`}>{sm.label}</p>

                <div className={`grid grid-cols-2 gap-4 mb-8 ${isDraft ? 'grayscale' : ''}`}>
                  <div className="bg-surface-container-low rounded-2xl p-4">
                    <p className="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mb-1">Candidats</p>
                    <p className="text-2xl font-black text-on-surface">{job.applicants_count ?? 0}</p>
                  </div>
                  <div className="bg-surface-container-low rounded-2xl p-4">
                    <p className="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mb-1">Score IA</p>
                    <p className={`text-2xl font-black ${job.avg_score != null ? ScoreColor(job.avg_score) : 'text-on-surface-variant'}`}>
                      {job.avg_score != null ? `${Math.round(job.avg_score)}%` : '—'}
                    </p>
                  </div>
                </div>

                <div className="mt-auto">
                  <div className="flex justify-between items-end mb-2">
                    <p className="text-xs font-bold text-on-surface-variant">Étape</p>
                    <p className={`text-xs font-bold ${isDraft ? 'text-on-surface-variant/40 italic' : 'text-primary'}`}>
                      {STAGE_LABEL[job.stage] ?? job.stage}
                    </p>
                  </div>
                  <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-primary-container rounded-full transition-all"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          })}

          {/* Create new card */}
          <button onClick={() => setModal({ mode: 'create' })}
            className="border-2 border-dashed border-outline-variant/50 rounded-3xl p-8 hover:border-primary/50 hover:bg-primary/5 transition-all flex flex-col items-center justify-center group min-h-[340px]">
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:bg-primary group-hover:text-white transition-all text-on-surface-variant">
              <Icon name="add" size={32} />
            </div>
            <span className="text-lg font-bold text-on-surface group-hover:text-primary transition-colors">Créer une offre</span>
            <span className="text-sm font-medium text-on-surface-variant mt-1">Démarrer un nouveau cycle de recrutement</span>
          </button>
        </div>
      )}
    </div>
  )
}
