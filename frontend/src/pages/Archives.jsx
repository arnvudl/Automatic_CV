import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { getCandidates, updateStatus, deleteCandidate } from '../lib/api'
import { Toast } from '../components/Toast'

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

      {/* Modale confirmation suppression */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center" onClick={() => setConfirmDelete(null)}>
          <div className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient-lg max-w-sm w-full mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-error"><Icon name="warning" size={28} /></span>
              <h3 className="text-lg font-bold text-on-surface">Supprimer définitivement ?</h3>
            </div>
            <p className="text-sm text-on-surface-variant mb-6">Cette action est irréversible.</p>
            <div className="flex gap-3 justify-end">
              <button onClick={() => setConfirmDelete(null)}
                className="px-5 py-2 rounded-xl bg-surface-container text-on-surface font-semibold text-sm">
                Annuler
              </button>
              <button onClick={() => handleDelete(confirmDelete)} disabled={!!deleting}
                className="px-5 py-2 rounded-xl bg-error text-white font-semibold text-sm disabled:opacity-50 flex items-center gap-2">
                <Icon name="delete_forever" size={16} /> Supprimer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-10">
        <button onClick={() => onNavigate('candidates')}
          className="flex items-center gap-2 text-on-surface-variant hover:text-on-surface text-sm font-medium mb-4 transition-colors">
          <Icon name="arrow_back" size={18} /> Retour aux candidats
        </button>
        <h1 className="text-5xl font-black text-on-surface tracking-tight">Archives</h1>
        <p className="text-on-surface-variant mt-2 font-medium">
          {candidates.length} candidat{candidates.length > 1 ? 's' : ''} archivé{candidates.length > 1 ? 's' : ''}
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-24 text-on-surface-variant gap-3">
          <Icon name="hourglass_empty" size={32} />
          <span className="font-medium">Chargement...</span>
        </div>
      ) : candidates.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-on-surface-variant gap-4">
          <Icon name="inventory_2" size={64} />
          <p className="text-lg font-bold text-on-surface">Aucun candidat archivé</p>
          <p className="text-sm">Les candidats archivés apparaîtront ici</p>
        </div>
      ) : (
        <div className="bg-surface-container-lowest rounded-3xl overflow-hidden shadow-ambient">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-container-low/50">
                {['Candidat', 'Score', 'Décision', 'Archivé le', ''].map(h => (
                  <th key={h} className="px-6 py-5 text-[11px] font-bold text-outline uppercase tracking-wider first:pl-8 last:pr-8">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {candidates.map((c, i) => {
                const pct  = Math.round((c.score ?? 0) * 100)
                const date = c.received_at
                  ? new Date(c.received_at).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' })
                  : '—'
                const decisionLabel = { invite: 'Invité', reject: 'Rejeté', eliminated: 'Éliminé' }[c.decision] ?? '—'
                const decisionColor = { invite: 'text-tertiary', reject: 'text-error', eliminated: 'text-on-surface-variant' }[c.decision] ?? 'text-on-surface-variant'
                return (
                  <tr key={c.candidate_id} className="hover:bg-surface-container-low/30 transition-colors group">
                    <td className="px-8 py-5">
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-xl ${AVATAR_COLORS[i % AVATAR_COLORS.length]} flex items-center justify-center font-bold text-xs flex-shrink-0`}>
                          {initials(c.name)}
                        </div>
                        <div>
                          <div className="font-bold text-on-surface">{c.name ?? 'Anonyme'}</div>
                          <div className="text-xs text-on-surface-variant">{c.sector ?? '—'}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <span className="text-sm font-black text-on-surface-variant">{pct}%</span>
                    </td>
                    <td className="px-6 py-5">
                      <span className={`text-sm font-bold ${decisionColor}`}>{decisionLabel}</span>
                    </td>
                    <td className="px-6 py-5 text-sm text-on-surface-variant">{date}</td>
                    <td className="px-8 py-5" onClick={e => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button onClick={() => onNavigate('profile', c.candidate_id)}
                          className="p-2 hover:bg-surface-container-high rounded-lg text-on-surface-variant" title="Voir le profil">
                          <Icon name="visibility" size={18} />
                        </button>
                        <button onClick={() => handleRestore(c.candidate_id)}
                          className="p-2 hover:bg-tertiary/10 rounded-lg text-on-surface-variant hover:text-tertiary transition-colors" title="Restaurer">
                          <Icon name="unarchive" size={18} />
                        </button>
                        <button onClick={() => setConfirmDelete(c.candidate_id)}
                          className="p-2 hover:bg-error-container rounded-lg text-on-surface-variant hover:text-error transition-colors" title="Supprimer définitivement">
                          <Icon name="delete_forever" size={18} />
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
