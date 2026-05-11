import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getCandidates, updateStatus, deleteCandidate } from '../lib/api'
import { Toast } from '../components/Toast'

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

export default function Archives({ onNavigate }) {
  const [candidates, setCandidates] = useState([])
  const [loading, setLoading]       = useState(true)
  const [toast, setToast]           = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [deleting, setDeleting]     = useState(null)

  useEffect(() => {
    getCandidates({ status: 'archived', limit: 200 })
      .then(setCandidates)
      .catch(() => setCandidates([]))
      .finally(() => setLoading(false))
  }, [])

  const handleRestore = async (id) => {
    try {
      await updateStatus(id, 'inbox')
      setCandidates(prev => prev.filter(c => c.candidate_id !== id))
      setToast({ message: 'Candidat restauré', type: 'success' })
    } catch {
      setToast({ message: 'Erreur lors de la restauration', type: 'error' })
    }
  }

  const handleDelete = async (id) => {
    setDeleting(id)
    try {
      await deleteCandidate(id)
      setCandidates(prev => prev.filter(c => c.candidate_id !== id))
      setToast({ message: 'Candidat supprimé définitivement', type: 'info' })
    } catch {
      setToast({ message: 'Erreur lors de la suppression', type: 'error' })
    } finally {
      setDeleting(null)
      setConfirmDelete(null)
    }
  }

  return (
    <div className="p-10 max-w-6xl mx-auto">
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {/* Confirm delete */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={() => setConfirmDelete(null)}>
          <div className="bg-card border border-border rounded-xl p-8 shadow-card-lg max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-destructive"><Icon name="warning" size={28} /></span>
              <h3 className="text-lg font-bold text-foreground">Supprimer définitivement ?</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-6">Cette action est irréversible.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)}
                className="px-5 py-2 rounded-lg bg-muted text-foreground font-semibold text-sm hover:bg-muted/80 transition-colors">
                Annuler
              </button>
              <button onClick={() => handleDelete(confirmDelete)} disabled={!!deleting}
                className="px-5 py-2 rounded-lg bg-destructive text-white font-semibold text-sm disabled:opacity-50 hover:opacity-90 transition-opacity flex items-center gap-2">
                <Icon name="delete_forever" size={16} /> Supprimer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-8">
        <button onClick={() => onNavigate('candidates')}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground text-sm font-medium mb-4 transition-colors">
          <Icon name="arrow_back" size={18} /> Retour aux candidats
        </button>
        <h1 className="text-3xl font-bold text-foreground tracking-tight">Archives</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          {candidates.length} candidat{candidates.length !== 1 ? 's' : ''} archivé{candidates.length !== 1 ? 's' : ''}
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-24 text-muted-foreground gap-3">
          <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
          <span className="font-medium">Chargement...</span>
        </div>
      ) : candidates.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-muted-foreground gap-4">
          <Icon name="inventory_2" size={56} />
          <p className="text-base font-bold text-foreground">Aucun candidat archivé</p>
          <p className="text-sm">Les candidats archivés apparaîtront ici</p>
        </div>
      ) : (
        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-card">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-muted/50 border-b border-border">
                {['Candidat', 'Score', 'Décision', 'Archivé le', ''].map(h => (
                  <th key={h} className="px-6 py-4 text-[11px] font-bold text-muted-foreground uppercase tracking-wider first:pl-8 last:pr-8">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {candidates.map((c, i) => {
                const pct  = Math.round((c.score ?? 0) * 100)
                const date = c.received_at
                  ? new Date(c.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
                  : '—'
                const decisionLabel = { invite: 'Invité', reject: 'Rejeté', eliminated: 'Éliminé' }[c.decision] ?? '—'
                const decisionColor = {
                  invite:     'text-success',
                  reject:     'text-destructive',
                  eliminated: 'text-muted-foreground',
                }[c.decision] ?? 'text-muted-foreground'
                return (
                  <tr key={c.candidate_id} className="hover:bg-muted/30 transition-colors group">
                    <td className="px-8 py-4">
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-lg ${AVATAR_COLORS[i % AVATAR_COLORS.length]} flex items-center justify-center font-bold text-xs flex-shrink-0`}>
                          {initials(c.name)}
                        </div>
                        <div>
                          <div className="font-semibold text-foreground">{c.name ?? 'Anonyme'}</div>
                          <div className="text-xs text-muted-foreground">{c.sector ?? '—'}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm font-bold text-muted-foreground">{pct}%</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-sm font-semibold ${decisionColor}`}>{decisionLabel}</span>
                    </td>
                    <td className="px-6 py-4 text-sm text-muted-foreground">{date}</td>
                    <td className="px-8 py-4" onClick={e => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onClick={() => onNavigate('profile', c.candidate_id)}
                          className="p-2 hover:bg-muted rounded-lg text-muted-foreground hover:text-foreground transition-colors" title="Voir le profil">
                          <Icon name="visibility" size={16} />
                        </button>
                        <button onClick={() => handleRestore(c.candidate_id)}
                          className="p-2 hover:bg-success/10 rounded-lg text-muted-foreground hover:text-success transition-colors" title="Restaurer">
                          <Icon name="unarchive" size={16} />
                        </button>
                        <button onClick={() => setConfirmDelete(c.candidate_id)}
                          className="p-2 hover:bg-destructive/10 rounded-lg text-muted-foreground hover:text-destructive transition-colors" title="Supprimer définitivement">
                          <Icon name="delete_forever" size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
