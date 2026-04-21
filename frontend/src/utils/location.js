// Catégorise la localisation d'un candidat par rapport à Luxembourg
const COUNTRY_PREFIXES = {
  '1': 'USA/Canada', '234': 'Nigeria', '31': 'Pays-Bas',
  '33': 'France', '351': 'Portugal', '353': 'Irlande',
  '39': 'Italie', '48': 'Pologne', '49': 'Allemagne', '91': 'Inde',
  '352': 'Luxembourg',
}

const CLOSE_EUROPE = new Set(['France', 'Allemagne', 'Pays-Bas', 'Belgique', 'Luxembourg'])
const EU_EUROPE    = new Set(['Irlande', 'Italie', 'Pologne', 'Portugal', 'Espagne', 'Autriche'])

export function getCountry(phone) {
  if (!phone || !String(phone).startsWith('+')) return 'Inconnu'
  const p = String(phone).slice(1)
  for (const len of [3, 2, 1]) {
    const prefix = p.slice(0, len)
    if (COUNTRY_PREFIXES[prefix]) return COUNTRY_PREFIXES[prefix]
  }
  return 'Autre'
}

export function getLocationCategory(phone) {
  const country = getCountry(phone)
  if (country === 'Luxembourg')            return { label: '🇱🇺 Luxembourg',     color: 'bg-green-100 text-green-700' }
  if (CLOSE_EUROPE.has(country))          return { label: '🇪🇺 Europe proche',   color: 'bg-blue-100 text-blue-700' }
  if (EU_EUROPE.has(country))             return { label: '🌍 Europe',           color: 'bg-indigo-100 text-indigo-700' }
  return                                         { label: '✈️ International',    color: 'bg-orange-100 text-orange-700' }
}

export function getScoreColor(score) {
  if (score >= 0.6) return 'text-green-600'
  if (score >= 0.4) return 'text-yellow-600'
  return 'text-red-500'
}

export function getScoreBg(score) {
  if (score >= 0.6) return 'bg-green-50 border-green-200'
  if (score >= 0.4) return 'bg-yellow-50 border-yellow-200'
  return 'bg-red-50 border-red-200'
}
