# Roadmap Frontend — Automatic CV → TeamTailor-like

> Référence : https://www.teamtailor.com/fr/all-features/
> Rédigé le : 2026-05-10
> Stack actuelle : React 19, Tailwind CSS, dnd-kit, recharts, @react-pdf/renderer, Vite

---

## État actuel ✅

| Domaine | Implémenté |
|---|---|
| Auth JWT | Login, token localStorage, AuthContext, déconnexion auto |
| Dashboard | Stats globales (recharts), KPIs |
| Offres (Jobs) | CRUD complet |
| Candidats | Liste avec filtres, profil détaillé, changement de statut |
| Calendrier | CRUD entretiens, vue mensuelle |
| Commentaires | Par candidat |
| Rapport PDF | Export via @react-pdf/renderer |
| Drag & drop | dnd-kit installé (non branché sur pipeline) |
| Temps réel | Hook SSE (useRealtime) |
| Scoring ML | Upload CV → score + décision |
| Paramètres | Page Settings |
| Archives | Candidats archivés |

---

## Dev local (sans n8n)

```bash
# API uniquement
docker compose up api -d

# Frontend hot-reload
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

Le proxy Vite redirige automatiquement `/auth`, `/candidates`, `/jobs`, `/interviews`, `/stats`, `/score`, `/comments`, `/events` vers `localhost:8000`.

---

## Phase 1 — Pipeline Kanban ✅ *(terminé)*

**Pourquoi :** C'est le cœur d'un ATS. Sans pipeline visuel, le produit n'est qu'une liste de candidats.

**Durée estimée :** 1-2 semaines

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/pages/Pipeline.jsx` | Vue principale Kanban par offre |
| `src/components/KanbanBoard.jsx` | Plateau avec colonnes drag & drop (dnd-kit) |
| `src/components/KanbanColumn.jsx` | Une colonne = une étape du pipeline |
| `src/components/KanbanCard.jsx` | Carte candidat avec score, photo, tags |

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/jobs/{job_id}/stages` | GET | Retourne les étapes du pipeline pour une offre |
| `/jobs/{job_id}/stages` | POST | Crée une étape custom |
| `/candidates/{id}/stage` | PATCH | Déplace un candidat vers une autre étape |

### Étapes par défaut
1. Inbox (candidatures reçues)
2. Screening (examen CV/score ML)
3. Entretien téléphonique
4. Entretien technique
5. Offre envoyée
6. Embauché
7. Refusé

### Modèle DB à ajouter
```python
class PipelineStage(Base):
    __tablename__ = "pipeline_stages"
    stage_id   = Column(String(64), primary_key=True)
    job_id     = Column(String(64), nullable=False, index=True)
    name       = Column(String(128), nullable=False)
    position   = Column(Integer, nullable=False)  # ordre d'affichage
    color      = Column(String(16), nullable=True)  # hex couleur colonne

# Ajouter sur Candidate :
stage_id = Column(String(64), nullable=True)  # étape courante
```

---

## Phase 2 — Fiches d'évaluation & Qualification

**Durée estimée :** 1 semaine

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/components/Scorecard.jsx` | Grille d'évaluation notée 1-5 par critère |
| `src/components/QualificationForm.jsx` | Questions pré-screening liées à l'offre |
| `src/pages/CandidateCompare.jsx` | Vue côte-à-côte de 2-3 profils |

### Features

- **Scorecards** : critères configurables par offre (technique, motivation, soft skills…), note 1-5, commentaire libre, synthèse par recruteur
- **Questions de qualification** : formulaire attaché à l'offre, réponses stockées sur le candidat, filtrage automatique si score < seuil
- **Comparaison candidats** : sélectionner 2-3 candidats depuis la liste → vue parallèle des scores, expérience, compétences
- **Candidats anonymes** : toggle pour masquer nom/photo (réduction de biais)

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/jobs/{job_id}/scorecard-template` | GET/PUT | Template de critères pour l'offre |
| `/candidates/{id}/scorecards` | GET/POST | Fiches remplies par les recruteurs |
| `/jobs/{job_id}/qualification-questions` | GET/PUT | Questions de pré-screening |

---

## Phase 3 — Communication candidats

**Durée estimée :** 1 semaine

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/pages/Messaging.jsx` | Boîte de messagerie par candidat |
| `src/components/MessageTemplates.jsx` | Bibliothèque de modèles |
| `src/components/BulkMessage.jsx` | Envoi groupé à une sélection |

### Features

- **Modèles de messages** : templates réutilisables (convocation, refus, relance…) avec variables `{{nom}}`, `{{poste}}`
- **Messagerie candidat** : timeline des échanges par candidat (email/SMS simulé en interne)
- **Messages groupés** : sélection multi-candidats depuis la liste → envoi unique
- **Campagnes d'accompagnement** : séquences automatiques J+3 / J+7 / J+14 configurables par étape

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/message-templates` | GET/POST/DELETE | CRUD modèles |
| `/messages/{candidate_id}` | GET/POST | Historique + envoi |
| `/messages/bulk` | POST | Envoi groupé |

---

## Phase 4 — IA Co-pilot

**Durée estimée :** 1 semaine
**Note :** Groq est déjà branché dans `api/main.py` (`llama-3.3-70b-versatile`).

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/components/AiJobWriter.jsx` | Génération de description d'offre |
| `src/components/AiCvSummary.jsx` | Résumé automatique du CV candidat |
| `src/components/AiInterviewQuestions.jsx` | Questions d'entretien suggérées |
| `src/components/AiCandidateSuggestions.jsx` | Candidats similaires dans le vivier |

### Features

- **Rédaction d'offres** : saisir titre + département → IA génère description complète
- **Résumé CV** : bouton sur profil candidat → 3-4 lignes de synthèse générées
- **Questions d'entretien** : basées sur le poste + le profil du candidat
- **Suggestions** : "Ces candidats existants correspondent à cette offre"
- **Comptes-rendus d'entretien** : notes IA depuis un résumé saisi à la main

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/ai/generate-job-description` | POST | `{title, department}` → texte |
| `/ai/summarize-cv/{candidate_id}` | GET | Résumé du profil |
| `/ai/interview-questions` | POST | `{job_id, candidate_id}` → liste |
| `/ai/suggest-candidates` | POST | `{job_id}` → candidats matchés |

---

## Phase 5 — Vivier de talents & Sourcing

**Durée estimée :** 1 semaine

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/pages/TalentPool.jsx` | Vivier de candidats passifs |
| `src/components/TagManager.jsx` | Gestion de tags libres |
| `src/components/ReferralForm.jsx` | Formulaire de cooptation |
| `src/components/SourceBadge.jsx` | Badge origine candidat (LinkedIn, Indeed…) |

### Features

- **Talent Pool** : vue dédiée aux candidats archivés mais disponibles pour futures offres
- **Tags** : labels libres sur candidats et offres, filtrables
- **Cooptations** : formulaire interne pour recommandations par des employés
- **Source tracking** : champ `source` sur chaque candidat (LinkedIn, Indeed, direct, cooptation…)
- **Réactivation** : notifier des candidats du vivier quand une offre correspondante s'ouvre

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/candidates/{id}/tags` | GET/POST/DELETE | Gestion tags candidat |
| `/talent-pool` | GET | Candidats avec statut `talent_pool` |
| `/referrals` | GET/POST | CRUD cooptations |

---

## Phase 6 — Analytics avancées

**Durée estimée :** 1 semaine

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/pages/Analytics.jsx` | Hub des rapports |
| `src/components/RecruitmentFunnel.jsx` | Entonnoir de conversion par étape |
| `src/components/JobReport.jsx` | Rapport par offre (time-to-hire, scores) |
| `src/components/UserReport.jsx` | Activité par recruteur |
| `src/components/NpsSurvey.jsx` | Enquête satisfaction candidat |

### Features

- **Funnel de recrutement** : taux de conversion Inbox → Screening → Entretien → Offre → Embauché
- **Rapports par offre** : time-to-hire, nb candidats, score moyen, source principale
- **Rapports utilisateurs** : candidats traités, entretiens réalisés, temps de réponse moyen
- **NPS candidat** : score de satisfaction envoyé en fin de process
- **Rapports historiques** : comparaison sur 3/6/12 mois

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/analytics/funnel` | GET | Conversion par étape |
| `/analytics/jobs/{job_id}` | GET | Rapport détaillé par offre |
| `/analytics/users` | GET | Activité par recruteur |
| `/analytics/time-to-hire` | GET | Durée moyenne par offre/période |

---

## Phase 7 — Onboarding & Admin

**Durée estimée :** 1 semaine

### Frontend à créer

| Fichier | Rôle |
|---|---|
| `src/pages/Onboarding.jsx` | Vue onboarding des nouvelles recrues |
| `src/components/OnboardingChecklist.jsx` | Checklist de tâches assignées |
| `src/pages/AdminUsers.jsx` | Gestion utilisateurs (admin seulement) |
| `src/pages/AuditLog.jsx` | Journal d'audit des actions |

### Features

- **Tâches d'onboarding** : checklist configurable (contrat, accès IT, badge…) assignée à la recrue
- **Page candidat personnelle** : portail read-only pour que la recrue suive son dossier
- **Gestion utilisateurs** : UI admin pour créer/désactiver/changer le rôle des utilisateurs
- **Droits d'accès** : Admin / Recruteur / Manager (accès restreint à certaines étapes)
- **Journal d'audit** : log horodaté de toutes les actions sensibles
- **RGPD** : suppression automatique des candidats après N mois configurables

### Backend à créer

| Endpoint | Méthode | Description |
|---|---|---|
| `/onboarding/{candidate_id}/tasks` | GET/POST/PATCH | CRUD tâches onboarding |
| `/audit-logs` | GET | Journal paginé (admin uniquement) |
| `/settings/rgpd` | GET/PUT | Configuration rétention des données |

---

## Ordre de développement recommandé

```
Phase 1 Kanban     → dnd-kit déjà là, impact UX maximal
Phase 2 Scorecards → différenciateur vs liste basique
Phase 4 IA         → Groq déjà branché, quick win
Phase 3 Messaging  → communication candidat
Phase 5 Vivier     → rétention des profils
Phase 6 Analytics  → recharts déjà là
Phase 7 Admin      → polish final
```

---

## Conventions de code

- Composants en PascalCase, hooks en `use*`
- Appels API uniquement via `src/lib/api.js`
- Pas de state management externe (React Context suffit pour l'auth)
- Tailwind pour le style, pas de CSS custom sauf `index.css`
- dnd-kit pour tout ce qui est drag & drop
- recharts pour tous les graphiques
