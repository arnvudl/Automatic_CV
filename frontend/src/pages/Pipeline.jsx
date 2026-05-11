import { useState, useEffect } from 'react'
import { Icon } from '../components/Icon'
import { KanbanBoard } from '../components/KanbanBoard'
import { getJobs, getPipelineStages, createPipelineStage } from '../lib/api'

export default function Pipeline({ onNavigate }) {
  const [jobs, setJobs]             = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [stages, setStages]         = useState([])
  const [loadingJobs, setLoadingJobs]   = useState(true)
  const [loadingBoard, setLoadingBoard] = useState(false)
  const [showAddStage, setShowAddStage] = useState(false)
  const [newStageName, setNewStageName] = useState('')
  const [addingStage, setAddingStage]   = useState(false)

  // Charger les offres
  useEffect(() => {
    getJobs({ status: 'active' })
      .then(data => {
        setJobs(data)
        if (data.length > 0) setSelectedJobId(data[0].job_id)
      })
      .catch(() => setJobs([]))
      .finally(() => setLoadingJobs(false))
  }, [])

  // Charger le pipeline quand l'offre change
  useEffect(() => {
    if (!selectedJobId) return
    setLoadingBoard(true)
    getPipelineStages(selectedJobId)
      .then(setStages)
      .catch(() => setStages([]))
      .finally(() => setLoadingBoard(false))
  }, [selectedJobId])

  const selectedJob = jobs.find(j => j.job_id === selectedJobId)

  const totalCandidates = stages.reduce((n, s) => n + s.candidates.length, 0)

  const handleAddStage = async () => {
    if (!newStageName.trim() || !selectedJobId) return
    setAddingStage(true)
    try {
      const newStage = await createPipelineStage(selectedJobId, { name: newStageName.trim() })
      setStages(prev => [...prev, { ...newStage, candidates: [] }])
      setNewStageName('')
      setShowAddStage(false)
    } catch (_) {}
    finally { setAddingStage(false) }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">     {/* 57px = hauteur TopNav */}

      {/* Toolbar */}
      <div className="flex items-center justify-between px-8 py-4 border-b border-border bg-card flex-shrink-0">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-lg font-bold text-foreground">Pipeline</h1>
            <p className="text-xs text-muted-foreground">
              {selectedJob ? selectedJob.title : 'Sélectionnez une offre'}
              {!loadingBoard && stages.length > 0 && ` · ${totalCandidates} candidat${totalCandidates !== 1 ? 's' : ''}`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Sélecteur d'offre */}
          {loadingJobs ? (
            <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
          ) : jobs.length === 0 ? (
            <button onClick={() => onNavigate('jobs')}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-primary-foreground rounded-lg text-sm font-semibold hover:opacity-90 transition-opacity">
              <Icon name="add" size={16} /> Créer une offre
            </button>
          ) : (
            <select
              value={selectedJobId ?? ''}
              onChange={e => setSelectedJobId(e.target.value)}
              className="bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 transition-all">
              {jobs.map(j => (
                <option key={j.job_id} value={j.job_id}>{j.title}</option>
              ))}
            </select>
          )}

          {/* Ajouter une colonne */}
          {selectedJobId && !showAddStage && (
            <button
              onClick={() => setShowAddStage(true)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-border text-sm font-semibold text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
              <Icon name="add" size={16} /> Colonne
            </button>
          )}
          {showAddStage && (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                value={newStageName}
                onChange={e => setNewStageName(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleAddStage(); if (e.key === 'Escape') setShowAddStage(false) }}
                placeholder="Nom de l'étape…"
                className="bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-foreground/20 w-40"
              />
              <button onClick={handleAddStage} disabled={addingStage || !newStageName.trim()}
                className="px-3 py-2 bg-foreground text-primary-foreground rounded-lg text-sm font-semibold disabled:opacity-50 hover:opacity-90 transition-opacity">
                {addingStage ? '…' : 'Ajouter'}
              </button>
              <button onClick={() => setShowAddStage(false)}
                className="p-2 text-muted-foreground hover:text-foreground transition-colors">
                <Icon name="close" size={16} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Board */}
      <div className="flex-1 overflow-auto p-6">
        {!selectedJobId ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-4">
            <Icon name="work_outline" size={48} />
            <div className="text-center">
              <p className="text-base font-semibold text-foreground">Aucune offre active</p>
              <p className="text-sm mt-1">Créez une offre d'emploi pour démarrer le pipeline</p>
            </div>
            <button onClick={() => onNavigate('jobs')}
              className="px-5 py-2.5 bg-foreground text-primary-foreground rounded-lg text-sm font-bold hover:opacity-90 transition-opacity flex items-center gap-2">
              <Icon name="add" size={16} /> Créer une offre
            </button>
          </div>
        ) : loadingBoard ? (
          <div className="flex items-center justify-center h-full gap-3 text-muted-foreground">
            <div className="w-5 h-5 border-2 border-border border-t-foreground rounded-full animate-spin" />
            <span className="text-sm">Chargement du pipeline…</span>
          </div>
        ) : stages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
            <Icon name="view_kanban" size={48} />
            <p className="text-sm">Aucune étape dans ce pipeline</p>
          </div>
        ) : (
          <KanbanBoard
            key={selectedJobId}        // remonte le board à chaque changement de job
            stages={stages}
            onNavigate={onNavigate}
          />
        )}
      </div>
    </div>
  )
}
