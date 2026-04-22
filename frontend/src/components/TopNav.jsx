import { Icon } from './Icon'

export default function TopNav({ onNavigate }) {
  return (
    <header className="bg-surface/80 backdrop-blur-xl sticky top-0 z-40 flex justify-between items-center w-full px-8 py-3">
      {/* Search */}
      <div className="flex items-center gap-8 flex-1">
        <div className="relative w-full max-w-md">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-outline">
            <Icon name="search" size={18} />
          </span>
          <input
            className="w-full bg-surface-container-high border-none rounded-xl py-2 pl-10 pr-4 text-sm outline-none focus:ring-2 focus:ring-primary/30 transition-all"
            placeholder="Search candidates, jobs, or tasks..."
          />
        </div>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-4">
        <button className="p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
          <Icon name="notifications" size={22} />
        </button>
        <button className="p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
          <Icon name="help_outline" size={22} />
        </button>

        <div className="h-8 w-px bg-outline-variant/30" />

        <button
          onClick={() => onNavigate('jobs')}
          className="px-5 py-2 bg-gradient-to-r from-primary to-primary-container text-white text-sm font-semibold rounded-full shadow-sm hover:shadow-md transition-all active:scale-95"
        >
          Create Job
        </button>

        <div className="w-9 h-9 rounded-full bg-secondary-container flex items-center justify-center text-primary font-bold text-sm border-2 border-white shadow-sm select-none">
          HR
        </div>
      </div>
    </header>
  )
}
