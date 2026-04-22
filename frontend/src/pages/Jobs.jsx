import { Icon } from '../components/Icon'

const JOBS = [
  { title: 'Senior Product Designer', dept: 'Experience Design', location: 'San Francisco, CA', applicants: 142, score: 92, stage: 'Final Interview', progress: 85, badge: 'High Priority', badgeColor: 'bg-tertiary-container/10 text-tertiary' },
  { title: 'Backend Infrastructure Engineer', dept: 'Engineering', location: 'Remote (Global)', applicants: 318, score: 78, stage: 'Technical Review', progress: 45, badge: 'Mid-Level', badgeColor: 'bg-secondary-container/20 text-on-secondary-container' },
  { title: 'Director of Performance Marketing', dept: 'Growth & Marketing', location: 'New York, NY', applicants: 54, score: 88, stage: 'Sourcing', progress: 15, badge: 'Strategic', badgeColor: 'bg-primary-container/10 text-primary' },
  { title: 'QA Automation Specialist', dept: 'Engineering', location: 'Austin, TX', applicants: 0, score: null, stage: 'Inactive', progress: 0, badge: 'Draft', badgeColor: 'bg-surface-container-highest text-on-surface-variant', draft: true },
]

function ScoreColor(score) {
  if (score >= 85) return 'text-tertiary'
  if (score >= 70) return 'text-on-secondary-container'
  return 'text-on-surface-variant'
}

export default function Jobs() {
  return (
    <div className="p-10 min-h-screen">
      {/* Header */}
      <div className="mb-12">
        <h1 className="text-6xl font-black text-on-surface tracking-tight leading-tight mb-2">Open Positions</h1>
        <p className="text-on-surface-variant text-lg max-w-2xl font-medium opacity-70">
          Review active recruitment cycles and high-potential talent matches powered by Luminary Insight AI.
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
        <div className="md:col-span-2 bg-surface-container-lowest p-8 rounded-3xl shadow-ambient relative overflow-hidden group">
          <div className="absolute -right-12 -top-12 w-48 h-48 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-colors" />
          <p className="text-sm font-bold text-primary tracking-widest uppercase mb-4">Total Active Pool</p>
          <div className="flex items-baseline gap-4">
            <span className="text-6xl font-black text-on-surface">1,284</span>
            <span className="text-tertiary font-bold flex items-center gap-1">
              <Icon name="trending_up" size={16} /> 12%
            </span>
          </div>
          <p className="text-on-surface-variant text-sm mt-4 font-medium">Across 18 open departments</p>
        </div>

        <div className="bg-surface-container-lowest p-8 rounded-3xl shadow-ambient">
          <p className="text-sm font-bold text-on-surface-variant/60 tracking-widest uppercase mb-4">Avg Match Score</p>
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full border-4 border-tertiary/20 border-t-tertiary flex items-center justify-center ai-glow">
              <span className="text-tertiary font-bold text-lg">84</span>
            </div>
            <span className="text-on-surface text-3xl font-black">%</span>
          </div>
          <p className="text-on-surface-variant text-sm mt-4 font-medium">Top Tier Performance</p>
        </div>

        <div className="bg-primary-container p-8 rounded-3xl shadow-lg text-on-primary-container">
          <p className="text-sm font-bold tracking-widest uppercase mb-4 opacity-80">Interviews Today</p>
          <div className="text-5xl font-black">12</div>
          <p className="text-sm opacity-70 mt-4 font-medium">Scheduled across all roles</p>
        </div>
      </div>

      {/* Jobs Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
        {JOBS.map((job, i) => (
          <div key={i} className={`bg-surface-container-lowest rounded-3xl p-8 shadow-ambient hover:shadow-ambient-lg transition-all group flex flex-col h-full cursor-pointer ${job.draft ? 'opacity-60' : ''}`}>
            <div className="flex justify-between items-start mb-6">
              <span className={`${job.badgeColor} text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-tighter`}>
                {job.badge}
              </span>
              <button className="text-on-surface-variant/40 hover:text-on-surface transition-colors">
                <Icon name="more_vert" size={20} />
              </button>
            </div>

            <h3 className="text-[1.375rem] font-bold text-on-surface leading-snug group-hover:text-primary transition-colors mb-1">
              {job.title}
            </h3>
            <p className="text-on-surface-variant text-sm font-medium mb-6">{job.dept} • {job.location}</p>

            <div className={`grid grid-cols-2 gap-4 mb-8 ${job.draft ? 'grayscale' : ''}`}>
              <div className="bg-surface-container-low rounded-2xl p-4">
                <p className="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mb-1">Applicants</p>
                <p className="text-2xl font-black text-on-surface">{job.applicants}</p>
              </div>
              <div className="bg-surface-container-low rounded-2xl p-4">
                <p className="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mb-1">AI Match Score</p>
                <p className={`text-2xl font-black ${job.score ? ScoreColor(job.score) : 'text-on-surface-variant'}`}>
                  {job.score ? `${job.score}%` : '—'}
                </p>
              </div>
            </div>

            <div className="mt-auto">
              <div className="flex justify-between items-end mb-2">
                <p className="text-xs font-bold text-on-surface-variant">Recruitment Stage</p>
                <p className={`text-xs font-bold ${job.draft ? 'text-on-surface-variant/40 italic' : 'text-primary'}`}>{job.stage}</p>
              </div>
              <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-primary-container rounded-full transition-all"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>
          </div>
        ))}

        {/* Create New */}
        <button className="border-2 border-dashed border-outline-variant/50 rounded-3xl p-8 hover:border-primary/50 hover:bg-primary/5 transition-all flex flex-col items-center justify-center group min-h-[340px]">
          <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:bg-primary group-hover:text-white transition-all text-on-surface-variant">
            <Icon name="add" size={32} />
          </div>
          <span className="text-lg font-bold text-on-surface group-hover:text-primary transition-colors">Create New Job</span>
          <span className="text-sm font-medium text-on-surface-variant mt-1">Start a new recruitment cycle</span>
        </button>
      </div>
    </div>
  )
}
