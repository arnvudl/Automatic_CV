import { Icon } from '../components/Icon'

const SKILLS = ['React v18', 'TypeScript', 'Next.js', 'Tailwind CSS', 'Redux Toolkit', 'Node.js', 'AWS']

const WORK_HISTORY = [
  { title: 'Senior Software Engineer', company: 'Stripe', period: '2021 — Present', duration: '3 years 2 months', desc: 'Leading the frontend architectural transition for the global checkout experience. Spearheading the adoption of server components and reducing bundle size by 42%.', active: true },
  { title: 'Software Engineer L4', company: 'Google', period: '2018 — 2021', duration: '3 years 6 months', desc: 'Contributed to the core UI library used by Google Workspace. Optimized rendering pipelines for complex data visualizations in Google Sheets.', active: false },
]

const NOTES = [
  { author: 'Marcus Chen', ago: '2 days ago', text: '"Had a quick intro call. Elena is very sharp. She mentioned she\'s looking for a role with more autonomy and a focus on design systems. Highly recommend moving to technical round."', highlight: true },
  { author: 'Sarah Miller', ago: '4 days ago', text: '"Verified her portfolio. The attention to detail in her UI components is exceptional. Clean code, very modular approach."', highlight: false },
]

export default function CandidateProfile({ onNavigate }) {
  return (
    <div className="p-10 max-w-7xl mx-auto">
      {/* Breadcrumb */}
      <div className="flex justify-between items-center mb-10">
        <button
          onClick={() => onNavigate('candidates')}
          className="flex items-center gap-2 text-on-surface-variant hover:text-on-surface transition-colors text-sm font-medium"
        >
          <Icon name="arrow_back" size={18} />
          Back to Senior Frontend Developer Candidates
        </button>
        <div className="flex gap-4">
          <button className="px-6 py-2.5 rounded-full bg-secondary-container text-on-secondary-container font-semibold text-sm hover:opacity-90 transition-all">
            Archive
          </button>
          <button className="px-8 py-2.5 rounded-full bg-gradient-to-r from-primary to-primary-container text-white font-bold text-sm shadow-lg shadow-primary/20 active:scale-95 transition-transform">
            Move to Interview
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left Column */}
        <div className="col-span-4 space-y-6">
          {/* Profile Summary */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 text-center shadow-ambient">
            <div className="w-32 h-32 rounded-full mx-auto mb-6 bg-gradient-to-br from-primary/20 to-primary-container/20 flex items-center justify-center text-4xl font-black text-primary ring-4 ring-surface-container-low">
              ER
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-on-surface">Elena Rodriguez</h2>
            <p className="text-primary font-medium mb-6">Senior Frontend Engineer</p>

            <div className="flex justify-center gap-3 mb-8 flex-wrap">
              <span className="px-3 py-1 bg-surface-container rounded-full text-xs font-bold text-on-surface-variant">San Francisco, CA</span>
              <span className="px-3 py-1 bg-surface-container rounded-full text-xs font-bold text-on-surface-variant">Remote Friendly</span>
            </div>

            <div className="space-y-2 text-left">
              {[
                { icon: 'mail', text: 'elena.rod@example.com' },
                { icon: 'phone', text: '+1 (555) 012-3456' },
                { icon: 'link', text: 'LinkedIn Profile', link: true },
              ].map(({ icon, text, link }) => (
                <div key={icon} className="flex items-center gap-3 p-3 rounded-lg hover:bg-surface-container-low transition-colors">
                  <span className="text-primary"><Icon name={icon} size={20} /></span>
                  {link ? <a href="#" className="text-sm text-primary font-bold hover:underline">{text}</a>
                    : <span className="text-sm text-on-surface">{text}</span>}
                </div>
              ))}
            </div>

            <button className="w-full mt-8 flex items-center justify-center gap-2 p-4 bg-surface-container-high rounded-xl text-on-surface font-bold text-sm hover:bg-surface-container-highest transition-colors">
              <Icon name="description" size={18} /> View Original Resume
            </button>
          </section>

          {/* Skills */}
          <section className="bg-surface-container-low rounded-2xl p-6 shadow-ambient">
            <h3 className="text-sm font-black text-on-surface uppercase tracking-widest mb-4">Core Competencies</h3>
            <div className="flex flex-wrap gap-2">
              {SKILLS.map(s => (
                <span key={s} className="px-3 py-1 bg-white text-primary text-xs font-bold rounded-lg border border-primary/10">{s}</span>
              ))}
            </div>
          </section>
        </div>

        {/* Right Column */}
        <div className="col-span-8 space-y-8">
          {/* AI Insights */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 ai-glow relative overflow-hidden shadow-ambient">
            <div className="absolute top-0 right-0 p-6">
              <div className="flex items-center gap-2 bg-tertiary-container/10 px-4 py-2 rounded-full">
                <span className="w-2 h-2 bg-tertiary rounded-full animate-pulse" />
                <span className="text-tertiary font-black text-xs uppercase tracking-tighter">98 Match Score</span>
              </div>
            </div>

            <div className="flex items-center gap-3 mb-8">
              <span className="text-tertiary"><Icon name="auto_awesome" fill size={30} /></span>
              <h3 className="text-xl font-extrabold text-on-surface">AI Insights & Suitability</h3>
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div className="p-5 bg-surface-container-low rounded-2xl">
                <p className="text-xs font-bold text-tertiary uppercase mb-2">Key Strength</p>
                <p className="text-sm text-on-surface font-medium leading-relaxed">Elena possesses deep expertise in React ecosystem. Her recent projects align perfectly with our current migration to Next.js 14.</p>
              </div>
              <div className="p-5 bg-surface-container-low rounded-2xl">
                <p className="text-xs font-bold text-tertiary uppercase mb-2">Pedigree</p>
                <p className="text-sm text-on-surface font-medium leading-relaxed">Significant tenure at high-growth organizations (Google, Stripe) indicates strong engineering discipline and scale experience.</p>
              </div>
              <div className="p-5 bg-surface-container-low rounded-2xl col-span-2">
                <p className="text-xs font-bold text-tertiary uppercase mb-2">Technical Verdict</p>
                <p className="text-sm text-on-surface font-medium leading-relaxed">Candidate demonstrates "Product-Minded" engineering. Analysis of her GitHub contributions shows a heavy focus on accessibility (a11y) and performance optimization, which are critical for our Q4 roadmap.</p>
              </div>
            </div>
          </section>

          {/* Work History */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
            <h3 className="text-xl font-extrabold text-on-surface mb-8">Work History</h3>
            <div className="space-y-10 relative before:absolute before:left-[11px] before:top-2 before:bottom-2 before:w-[2px] before:bg-surface-container">
              {WORK_HISTORY.map((w, i) => (
                <div key={i} className="relative pl-12">
                  <div className={`absolute left-0 top-1 w-6 h-6 rounded-full flex items-center justify-center ring-4 ring-white
                    ${w.active ? 'bg-primary' : 'bg-surface-container-highest'}`}>
                    <span className="w-2 h-2 bg-white rounded-full" />
                  </div>
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h4 className="text-lg font-bold text-on-surface">{w.title}</h4>
                      <p className="text-sm font-medium text-primary">{w.company} • {w.period}</p>
                    </div>
                    <span className="text-[10px] font-black text-slate-400 uppercase">{w.duration}</span>
                  </div>
                  <p className="text-sm text-on-surface-variant leading-relaxed">{w.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Internal Notes */}
          <section className="bg-surface-container-lowest rounded-2xl p-8 shadow-ambient">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-extrabold text-on-surface">Internal Notes</h3>
              <button className="flex items-center gap-2 text-primary font-bold text-sm hover:opacity-80">
                <Icon name="add" size={18} /> Add Note
              </button>
            </div>
            <div className="space-y-4">
              {NOTES.map((n, i) => (
                <div key={i} className={`p-6 bg-surface-container-low rounded-2xl border-l-4 ${n.highlight ? 'border-primary' : 'border-outline-variant/30'}`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-black text-on-surface">{n.author}</span>
                      <span className="text-[10px] text-on-surface-variant">• {n.ago}</span>
                    </div>
                    <Icon name="more_horiz" size={18} className="text-outline" />
                  </div>
                  <p className="text-sm text-on-surface-variant italic leading-relaxed">{n.text}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
