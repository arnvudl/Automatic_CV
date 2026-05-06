import { useEffect } from 'react'
import { Icon } from './Icon'

export function Toast({ message, type = 'success', onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3000)
    return () => clearTimeout(t)
  }, [onClose])

  const styles = {
    success: 'bg-tertiary text-white',
    error:   'bg-error text-white',
    info:    'bg-primary text-white',
  }

  const icons = { success: 'check_circle', error: 'error', info: 'info' }

  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-3 px-5 py-3 rounded-2xl shadow-ambient-lg font-medium text-sm animate-in ${styles[type]}`}>
      <Icon name={icons[type]} size={20} />
      {message}
    </div>
  )
}
