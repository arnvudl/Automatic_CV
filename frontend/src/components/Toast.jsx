import { useEffect } from 'react'
import { Icon } from './Icon'

export function Toast({ message, type = 'success', onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3000)
    return () => clearTimeout(t)
  }, [onClose])

  const styles = {
    success: 'bg-success text-white',
    error:   'bg-destructive text-white',
    info:    'bg-foreground text-primary-foreground',
  }

  const icons = { success: 'check_circle', error: 'error', info: 'info' }

  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 px-5 py-3 rounded-xl shadow-card-lg font-medium text-sm ${styles[type]}`}>
      <Icon name={icons[type]} size={18} />
      {message}
    </div>
  )
}
