import { useEffect, useState } from 'react'
import { X, Mail, Phone, Briefcase, Globe, FileText, Code, MessageSquare, Send, Pencil, Trash2 } from 'lucide-react'
import { getCountry, getLocationCategory } from '../utils/location'

const EDU_LABELS = { 1: 'Bac ou moins', 2: 'Bachelor', 3: 'Master', 4: 'PhD' }
const MODEL_VER  = 'v3-Fairness-Aware'
const HR_USER    = 'RH'

function buildTemplate(type, candidate) {
  const name = candidate.name || 'Madame/Monsieur'
  const role = candidate.target_role || 'le poste'
  switch (type) {
    case 'received': return {
      subject: `LuxTalent – Votre candidature pour ${role} a bien été reçue`,
      body: `Bonjour ${name},\n\nNous avons bien reçu votre candidature pour le poste de ${role} et nous vous remercions de l'intérêt que vous portez à LuxTalent Advisory Group.\n\nVotre dossier est en cours d'examen par notre équipe RH. Nous reviendrons vers vous dans les meilleurs délais.\n\nCordialement,\nL'équipe RH – LuxTalent Advisory Group`,
    }
    case 'invite': return {
      subject: `LuxTalent – Invitation à un entretien pour ${role}`,
      body: `Bonjour ${name},\n\nAprès examen de votre candidature, nous avons le plaisir de vous inviter à un entretien pour le poste de ${role}.\n\nNous vous contacterons prochainement pour convenir d'un créneau.\n\nCordialement,\nL'équipe RH – LuxTalent Advisory Group`,
    }
    case 'reject': return {
      subject: `LuxTalent – Suite à votre candidature pour ${role}`,
      body: `Bonjour ${name},\n\nNous vous remercions d'avoir postulé pour ${role}. Après examen, votre profil ne correspond pas aux critères recherchés à ce stade.\n\nNous conservons votre dossier et reviendrons vers vous si une opportunité correspondante se présente.\n\nCordialement,\nL'équipe RH – LuxTalent Advisory Group`,
    }
    default: return { subject: '', body: '' }
  }
}

export default function CandidateDetail({ candidate: c, onClose, onAction }) {
  const [detail,     setDetail]     = useState(null)
  const [explain,    setExplain]    = useState(null)
  const [semantic,   setSemantic]   = useState(null)
  const [semLoading, setSemLoading] = useState(false)
  const [semError,   setSemError]   = useState(null)
  const [comments,   setComments]   = useState([])
  const [newComment, setNewComment] = useState('')
  const [editId,     setEditId]     = useState(null)
  const [editText,   setEditText]   = useState('')
  const [mailOpen,   setMailOpen]   = useState(null)
  const [activeTab,  setActiveTab]  = useState('profile')

  useEffect(() => {
    if (!c?.candidate_id) return
    setDetail(null); setExplain(null); setSemantic(null); setSemError(null)
    fetch(`/candidates/${c.candidate_id}`).then(r => r.ok ? r.json() : null).then(setDetail).catch(() => {})
    fetch(`/candidates/${c.candidate_id}/explain`).then(r => r.ok ? r.json() : null).then(setExplain).catch(() => {})
    fetch(`/comments/${c.candidate_id}`).then(r => r.json()).then(setComments).catch(() => setComments([]))
  }, [c?.candidate_id])

  const loadSemantic = () => {
    if (semantic || semLoading) return
    setSemLoading(true); setSemError(null)
    fetch(`/candidates/${c.candidate_id}/semantic`)
      .then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e.detail || 'Erreur')))
      .then(d => { setSemantic(d); setSemLoading(false) })
      .catch(e => { setSemError(String(e)); setSemLoading(false) })
  }

  useEffect(() => { if (activeTab === 'semantic') loadSemantic() }, [activeTab])

  if (!c) return null
  const loc   = getLocationCategory(c.phone)
  const country = getCountry(c.phone)
  const score = parseFloat(c.score) || 0
  const pct   = Math.round(score * 100)
  const thr   = parseFloat(c.threshold_used) || 0.5
  const gap   = Math.abs(score - thr)
  const isBorderline = gap <= 0.08

  const addComment = async () => {
    if (!newComment.trim()) return
    const created = await fetch(`/comments/${c.candidate_id}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author: HR_USER, text: newComment }),
    }).then(r => r.json()).catch(() => null)
    if (created) { setComments(p => [...p, created]); setNewComment('') }
  }

  const saveEdit = async (id) => {
    const updated = await fetch(`/comments/${c.candidate_id}/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: editText }),
    }).then(r => r.json()).catch(() => null)
    if (updated) { setComments(p => p.map(cm => cm.id === id ? updated : cm)); setEditId(null) }
  }

  const deleteComment = async (id) => {
    await fetch(`/comments/${c.candidate_id}/${id}`, { method: 'DELETE' }).catch(() => {})
    setComments(p => p.filter(cm => cm.id !== id))
  }

  const TABS = [
    { id: 'profile',  label: 'Profil' },
    { id: 'ia',       label: `IA${isBorderline ? ' ⚠' : ''}` },
    { id: 'semantic', label: '🧠 Sémantique' },
    { id: 'comments', label: `Notes${comments.length ? ` (${comments.length})` : ''}` },
    { id: 'contact',  label: 'Contacter' },
  ]

  return (
    <div className="fixed inset-0 z-50 flex" onClick={onClose}>
      <div className="flex-1 bg-slate-900/30 backdrop-blur-sm" />
      <div className="w-[520px] bg-white h-full overflow-y-auto border-l border-slate-200 shadow-2xl"
           onClick={e => e.stopPropagation()}>

        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 sticky top-0 z-10 bg-white/95 backdrop-blur">
          <h2 className="font-bold text-slate-800 text-sm">Fiche candidat</h2>
          <button onClick={onClose} className="p-1.5 hover:bg-slate-100 rounded-full transition-colors">
            <X size={16} className="text-slate-500" />
          </button>
        </div>

        <div className="p-5 space-y-4">

          {/* Identity */}
          <div className="flex items-start gap-3">
            <div className="w-11 h-11 rounded-2xl bg-blue-600 flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
              {c.name ? c.name.charAt(0) : '?'}
            </div>
            <div className="flex-1">
              <h3 className="text-base font-bold text-slate-800">{c.name || '—'}</h3>
              <p className="text-slate-400 text-xs mt-0.5">{c.target_role || '—'} · {c.sector || '—'}</p>
              <div className="flex gap-1.5 mt-1.5 flex-wrap">
                <span className={`text-xs px-2 py-0.5 rounded-full ${loc.color}`}>{loc.label}</span>
                {c.gender && <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">{c.gender}</span>}
                {c.age    && <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">{c.age} ans</span>}
                {isBorderline && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-semibold">
                    Borderline
                  </span>
                )}
              </div>
            </div>
            <div className="text-right flex-shrink-0">
              <div className={`text-2xl font-bold ${score >= 0.6 ? 'text-emerald-600' : score >= 0.4 ? 'text-amber-600' : 'text-red-500'}`}>
                {pct}%
              </div>
              <div className={`text-xs font-medium ${c.decision === 'invite' ? 'text-emerald-600' : 'text-red-500'}`}>
                {c.decision === 'invite' ? 'Invité' : 'Rejeté'}
              </div>
            </div>
          </div>

          {/* Score bar */}
          <div>
            <div className="w-full bg-slate-100 rounded-full h-2 relative">
              <div className={`h-2 rounded-full ${score >= 0.6 ? 'bg-emerald-500' : score >= 0.4 ? 'bg-amber-500' : 'bg-red-500'}`}
                   style={{ width: `${pct}%` }} />
              <div className="absolute top-0 h-2 w-0.5 bg-slate-400 rounded"
                   style={{ left: `${Math.round(thr * 100)}%` }} />
            </div>
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>0%</span>
              <span>Barre de sélection : {Math.round(thr * 100)}%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Tab bar */}
          <div className="flex gap-0.5 bg-slate-100 rounded-xl p-1">
            {TABS.map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                className={`flex-1 text-xs py-1.5 rounded-lg font-medium transition-all
                  ${activeTab === t.id ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}>
                {t.label}
              </button>
            ))}
          </div>

          {/* ── Profile ── */}
          {activeTab === 'profile' && (
            <div className="space-y-4">
              {detail?.summary && (
                <Section title="Résumé" icon={<FileText size={12}/>}>
                  <p className="text-xs text-slate-500 leading-relaxed">{detail.summary}</p>
                </Section>
              )}
              {detail && (detail.skills_tech?.length > 0 || detail.skills_meth?.length > 0) && (
                <Section title="Compétences" icon={<Code size={12}/>}>
                  {detail.skills_tech?.length > 0 && <SkillGroup label="Techniques" skills={detail.skills_tech} color="bg-blue-50 text-blue-700 border border-blue-100" />}
                  {detail.skills_meth?.length > 0 && <SkillGroup label="Méthodes"   skills={detail.skills_meth} color="bg-purple-50 text-purple-700 border border-purple-100" />}
                  {detail.skills_mgmt?.length > 0 && <SkillGroup label="Management" skills={detail.skills_mgmt} color="bg-slate-100 text-slate-600" />}
                </Section>
              )}
              {detail && (detail.languages?.length > 0 || detail.certifications?.length > 0) && (
                <Section title="Langues & Certifications" icon={<MessageSquare size={12}/>}>
                  {detail.languages?.length > 0 && (
                    <div className="mb-2">
                      <p className="text-xs text-slate-400 mb-1">Langues</p>
                      <div className="flex flex-wrap gap-1.5">
                        {detail.languages.map((l, i) => (
                          <span key={i} className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-100 px-2 py-0.5 rounded-full">{l}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {detail.certifications?.length > 0 && (
                    <div>
                      <p className="text-xs text-slate-400 mb-1">Certifications</p>
                      <div className="flex flex-wrap gap-1.5">
                        {detail.certifications.map((cert, i) => (
                          <span key={i} className="text-xs bg-amber-50 text-amber-700 border border-amber-100 px-2 py-0.5 rounded-full">{cert}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </Section>
              )}
              <Section title="Profil professionnel" icon={<Briefcase size={12}/>}>
                <Row label="Expérience"  value={c.years_experience ? `${c.years_experience} ans` : '—'} />
                <Row label="Formation"   value={EDU_LABELS[parseInt(c.education_level)] || '—'} />
                <Row icon={<Globe size={11}/>}  label="Pays"  value={country} />
                <Row icon={<Mail size={11}/>}   label="Email" value={c.email} />
                <Row icon={<Phone size={11}/>}  label="Tél."  value={c.phone} />
              </Section>
              <Section title="Trace d'audit" icon={null}>
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 space-y-1.5 text-xs font-mono">
                  <AuditLine label="Reçu"     value={c.received_at ? new Date(c.received_at).toLocaleString('fr-FR') : '—'} />
                  <AuditLine label="Modèle"   value={`${MODEL_VER} · Barre ${Math.round(thr*100)}%`} />
                  <AuditLine label="Source"   value={c.source_filename || '—'} />
                  <AuditLine label="Statut"   value={c.status || 'inbox'} />
                  <AuditLine label="Décision" value={`${c.decision === 'invite' ? 'Inviter' : 'Rejeter'} (${pct}%)`} />
                </div>
              </Section>
              <div className="space-y-2 pt-1">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Action RH</p>
                <div className="flex gap-2">
                  <ActionBtn label="Envoyer en entretien" color="bg-emerald-600 hover:bg-emerald-700 text-white"
                    onClick={() => { onAction(c.candidate_id, 'interview'); onClose() }} />
                  <ActionBtn label="Rejeter" color="bg-red-500 hover:bg-red-600 text-white"
                    onClick={() => { onAction(c.candidate_id, 'rejected'); onClose() }} />
                </div>
                <ActionBtn label="Mettre en revue" color="bg-amber-500 hover:bg-amber-600 text-white w-full"
                  onClick={() => { onAction(c.candidate_id, 'review'); onClose() }} />
                <ActionBtn label="Remettre dans l'inbox" color="bg-slate-200 hover:bg-slate-300 text-slate-700 w-full"
                  onClick={() => { onAction(c.candidate_id, 'inbox'); onClose() }} />
              </div>
            </div>
          )}

          {/* ── IA ── */}
          {activeTab === 'ia' && (
            <div className="space-y-4">
              {!explain ? (
                <p className="text-slate-400 text-sm text-center py-8">Chargement…</p>
              ) : (
                <>
                  <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                    <p className="text-xs font-semibold text-blue-600 mb-1">Résumé IA</p>
                    <p className="text-sm text-slate-700 leading-relaxed">{explain.narrative}</p>
                  </div>
                  {isBorderline && (
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-xs text-amber-700">
                      Score à {Math.round(gap * 100)} pts de la barre de sélection — révision manuelle recommandée.
                    </div>
                  )}
                  {Object.keys(explain.shap || {}).length > 0 && (
                    <Section title="Impact par variable (SHAP)" icon={null}>
                      <ShapBars shap={explain.shap} />
                    </Section>
                  )}
                  {explain.missing?.length > 0 && (
                    <Section title="Ce qui lui manque vs profils invités" icon={null}>
                      <div className="space-y-2">
                        {explain.missing.map((m, i) => (
                          <div key={i} className="bg-red-50 border border-red-100 rounded-lg p-2.5">
                            <div className="flex justify-between items-center mb-1">
                              <span className="text-xs font-medium text-red-700">{m.feature}</span>
                              <span className="text-xs text-red-500 font-semibold">-{m.gap_pct}%</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-red-100 h-1.5 rounded-full overflow-hidden">
                                <div className="h-full bg-red-400 rounded-full"
                                     style={{ width: `${Math.min(m.candidate_value / m.invited_avg * 100, 100)}%` }} />
                              </div>
                              <span className="text-xs text-slate-400 w-28 text-right">
                                {m.candidate_value} vs {m.invited_avg}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Section>
                  )}
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-400">
                    Barre de sélection : {Math.round(thr * 100)}% · Confiance : {
                      score >= 0.75 ? 'Haute' : score >= 0.5 ? 'Moyenne' : score >= 0.35 ? 'Faible' : 'Très faible'
                    } · {MODEL_VER}
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Sémantique ── */}
          {activeTab === 'semantic' && (
            <div className="space-y-4">
              {semLoading && (
                <div className="flex flex-col items-center justify-center py-12 gap-3 text-slate-400">
                  <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  <p className="text-sm">Analyse Claude en cours…</p>
                </div>
              )}
              {semError && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-xs text-red-700">
                  {semError}
                </div>
              )}
              {semantic && !semLoading && (
                <>
                  {/* Score + recommandation */}
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center">
                      <span className="text-white font-bold text-xl">{semantic.semantic_score}</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-xs font-semibold text-violet-600 mb-0.5">Score sémantique Claude</p>
                      <p className="text-xs text-slate-600 leading-relaxed italic">"{semantic.recommendation}"</p>
                    </div>
                  </div>

                  {/* Trajectoire */}
                  <Section title="Trajectoire & sens de carrière" icon={null}>
                    <p className="text-xs text-slate-600 leading-relaxed">{semantic.trajectory}</p>
                  </Section>

                  {/* Équivalences compétences */}
                  {semantic.skill_equivalencies?.length > 0 && (
                    <Section title="Équivalences compétences" icon={null}>
                      <div className="space-y-2">
                        {semantic.skill_equivalencies.map((s, i) => (
                          <div key={i} className="flex items-center gap-2 text-xs">
                            <span className="text-slate-500 flex-1">{s.stated}</span>
                            <span className="text-slate-300">→</span>
                            <span className="font-medium text-slate-700 flex-1">{s.equivalent}</span>
                            <span className={`px-1.5 py-0.5 rounded-full text-xs font-medium
                              ${s.level === 'expert' ? 'bg-violet-100 text-violet-700'
                              : s.level === 'confirmed' ? 'bg-blue-100 text-blue-700'
                              : 'bg-slate-100 text-slate-500'}`}>
                              {s.level}
                            </span>
                          </div>
                        ))}
                      </div>
                    </Section>
                  )}

                  {/* Hidden gems */}
                  {semantic.hidden_gems?.length > 0 && (
                    <Section title="Signaux positifs cachés" icon={null}>
                      <div className="space-y-1.5">
                        {semantic.hidden_gems.map((g, i) => (
                          <div key={i} className="flex gap-2 items-start bg-emerald-50 border border-emerald-100 rounded-lg p-2 text-xs text-emerald-700">
                            <span>✦</span><span>{g}</span>
                          </div>
                        ))}
                      </div>
                    </Section>
                  )}

                  {/* Gaps de carrière */}
                  {semantic.career_gaps?.length > 0 && (
                    <Section title="Interruptions de parcours" icon={null}>
                      <div className="space-y-2">
                        {semantic.career_gaps.map((g, i) => (
                          <div key={i} className={`rounded-xl p-3 border text-xs space-y-0.5
                            ${g.impact === 'fort' ? 'bg-amber-50 border-amber-200'
                            : g.impact === 'modéré' ? 'bg-yellow-50 border-yellow-200'
                            : 'bg-slate-50 border-slate-200'}`}>
                            <div className="flex justify-between">
                              <span className="font-medium text-slate-700">{g.period}</span>
                              <span className={`px-1.5 py-0.5 rounded-full font-medium
                                ${g.impact === 'fort' ? 'bg-amber-200 text-amber-800'
                                : g.impact === 'modéré' ? 'bg-yellow-200 text-yellow-800'
                                : 'bg-slate-200 text-slate-600'}`}>
                                {g.impact}
                              </span>
                            </div>
                            <p className="text-slate-500">{g.likely_reason}</p>
                          </div>
                        ))}
                      </div>
                    </Section>
                  )}

                  {/* Red flags */}
                  {semantic.red_flags?.length > 0 && (
                    <Section title="Points de vigilance" icon={null}>
                      <div className="space-y-1.5">
                        {semantic.red_flags.map((f, i) => (
                          <div key={i} className="flex gap-2 items-start bg-red-50 border border-red-100 rounded-lg p-2 text-xs text-red-700">
                            <span>⚠</span><span>{f}</span>
                          </div>
                        ))}
                      </div>
                    </Section>
                  )}

                  <p className="text-xs text-slate-300 text-center">Analyse par {semantic.model}</p>
                </>
              )}
            </div>
          )}

          {/* ── Comments ── */}
          {activeTab === 'comments' && (
            <div className="space-y-3">
              {comments.length === 0 && (
                <p className="text-slate-300 text-sm text-center py-6">Aucun commentaire.</p>
              )}
              {comments.map(cm => (
                <div key={cm.id} className="bg-slate-50 border border-slate-200 rounded-xl p-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold text-white">
                        {cm.author.charAt(0)}
                      </div>
                      <span className="text-xs font-semibold text-slate-700">{cm.author}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">
                        {new Date(cm.created_at).toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}
                        {cm.updated_at && ' (modifié)'}
                      </span>
                      <button onClick={() => { setEditId(cm.id); setEditText(cm.text) }}
                        className="p-0.5 hover:text-blue-500 text-slate-300 transition-colors">
                        <Pencil size={11} />
                      </button>
                      <button onClick={() => deleteComment(cm.id)}
                        className="p-0.5 hover:text-red-500 text-slate-300 transition-colors">
                        <Trash2 size={11} />
                      </button>
                    </div>
                  </div>
                  {editId === cm.id ? (
                    <div className="flex gap-2 mt-2">
                      <textarea value={editText} onChange={e => setEditText(e.target.value)}
                        className="glass-input flex-1 text-xs p-2 resize-none h-16 text-slate-700" />
                      <div className="flex flex-col gap-1">
                        <button onClick={() => saveEdit(cm.id)} className="glass-btn-primary text-xs px-3 py-1">Sauver</button>
                        <button onClick={() => setEditId(null)} className="glass-btn-ghost text-xs px-3 py-1">Annuler</button>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-600 leading-relaxed">{cm.text}</p>
                  )}
                </div>
              ))}
              <div className="flex gap-2 pt-1">
                <textarea value={newComment} onChange={e => setNewComment(e.target.value)}
                  placeholder="Ajouter un commentaire…"
                  className="glass-input flex-1 text-xs p-3 resize-none h-16 text-slate-700"
                  onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) addComment() }} />
                <button onClick={addComment} className="glass-btn-primary px-3 flex items-center justify-center">
                  <Send size={14} />
                </button>
              </div>
              <p className="text-xs text-slate-300">Cmd+Entrée pour envoyer</p>
            </div>
          )}

          {/* ── Contact ── */}
          {activeTab === 'contact' && (
            <div className="space-y-3">
              <p className="text-xs text-slate-400">
                Sélectionner un template — l'email s'ouvrira dans votre client mail avec le texte pré-rempli.
              </p>
              {[
                { type: 'received', label: 'CV bien reçu et en cours de lecture', color: 'border-blue-200 hover:bg-blue-50' },
                { type: 'invite',   label: 'Invitation à un entretien',            color: 'border-emerald-200 hover:bg-emerald-50' },
                { type: 'reject',   label: 'Retour négatif',                        color: 'border-red-200 hover:bg-red-50' },
              ].map(({ type, label, color }) => (
                <button key={type} onClick={() => setMailOpen(mailOpen === type ? null : type)}
                  className={`w-full text-left p-3 rounded-xl border bg-white transition-all ${color}`}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-700 font-medium">{label}</span>
                    <Mail size={13} className="text-slate-400" />
                  </div>
                </button>
              ))}
              {mailOpen && (() => {
                const tpl    = buildTemplate(mailOpen, c)
                const mailto = `mailto:${c.email || ''}?subject=${encodeURIComponent(tpl.subject)}&body=${encodeURIComponent(tpl.body)}`
                return (
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
                    <div>
                      <p className="text-xs text-slate-400 mb-1">Objet</p>
                      <p className="text-xs text-slate-800 font-medium">{tpl.subject}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-1">Corps</p>
                      <pre className="text-xs text-slate-600 whitespace-pre-wrap leading-relaxed font-sans">{tpl.body}</pre>
                    </div>
                    <div className="flex gap-2 pt-1">
                      <a href={mailto} className="glass-btn-primary text-xs px-4 py-2 flex items-center gap-1.5">
                        <Mail size={12} /> Ouvrir dans le client mail
                      </a>
                      <button onClick={() => navigator.clipboard.writeText(`Objet: ${tpl.subject}\n\n${tpl.body}`)}
                        className="glass-btn-ghost text-xs px-4 py-2">
                        Copier
                      </button>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}

        </div>
      </div>
    </div>
  )
}

function Section({ title, icon, children }) {
  return (
    <div>
      {title && (
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-1">
          {icon}{title}
        </p>
      )}
      <div className="space-y-1.5">{children}</div>
    </div>
  )
}

function Row({ icon, label, value }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-400 flex items-center gap-1.5">{icon}{label}</span>
      <span className="font-medium text-slate-700 text-right max-w-[240px] truncate">{value || '—'}</span>
    </div>
  )
}

function AuditLine({ label, value }) {
  return (
    <div className="flex gap-2 text-slate-500">
      <span className="text-slate-400 w-16 shrink-0">{label}</span>
      <span>{value}</span>
    </div>
  )
}

function SkillGroup({ label, skills, color }) {
  return (
    <div className="mb-2">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <div className="flex flex-wrap gap-1">
        {skills.map((s, i) => (
          <span key={i} className={`text-xs px-2 py-0.5 rounded-full ${color}`}>{s}</span>
        ))}
      </div>
    </div>
  )
}

function ShapBars({ shap }) {
  const entries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 7)
  const maxAbs  = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001)
  return (
    <div className="space-y-2">
      {entries.map(([name, val]) => (
        <div key={name}>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-slate-500">{name}</span>
            <span className={val >= 0 ? 'text-emerald-600 font-medium' : 'text-red-500 font-medium'}>
              {val >= 0 ? '+' : ''}{val.toFixed(3)}
            </span>
          </div>
          <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${val >= 0 ? 'bg-emerald-400' : 'bg-red-400'}`}
                 style={{ width: `${Math.abs(val) / maxAbs * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function ActionBtn({ label, color, onClick }) {
  return (
    <button onClick={onClick}
      className={`text-sm font-medium px-4 py-2 rounded-xl transition-all flex-1 ${color}`}>
      {label}
    </button>
  )
}
