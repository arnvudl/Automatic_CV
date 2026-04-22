// Thin wrapper for Material Symbols Outlined
export function Icon({ name, fill = false, size = 24, className = '' }) {
  return (
    <span
      className={fill ? 'material-symbols-filled' : 'material-symbols-outlined'}
      style={{ fontSize: size }}
      aria-hidden
    >
      {name}
    </span>
  )
}
