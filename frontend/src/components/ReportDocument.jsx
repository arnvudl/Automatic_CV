import {
  Document, Page, Text, View, StyleSheet, Font
} from '@react-pdf/renderer'

const S = StyleSheet.create({
  page: {
    fontFamily: 'Helvetica',
    fontSize: 10,
    color: '#1e293b',
    padding: 40,
    backgroundColor: '#ffffff',
  },
  header: {
    marginBottom: 20,
    borderBottom: '2px solid #2563EB',
    paddingBottom: 10,
  },
  title: { fontSize: 18, fontFamily: 'Helvetica-Bold', color: '#1e3a8a' },
  subtitle: { fontSize: 10, color: '#64748b', marginTop: 3 },
  section: { marginTop: 18 },
  sectionTitle: {
    fontSize: 12, fontFamily: 'Helvetica-Bold', color: '#1e3a8a',
    marginBottom: 6, borderBottom: '1px solid #e2e8f0', paddingBottom: 3,
  },
  kpiRow: { flexDirection: 'row', gap: 12, marginBottom: 12 },
  kpi: {
    flex: 1, padding: 10, backgroundColor: '#f8fafc',
    borderRadius: 6, borderLeft: '3px solid #2563EB',
  },
  kpiLabel: { fontSize: 8, color: '#94a3b8', marginBottom: 2 },
  kpiValue: { fontSize: 16, fontFamily: 'Helvetica-Bold', color: '#1e293b' },
  row: { flexDirection: 'row', borderBottom: '1px solid #f1f5f9', paddingVertical: 5 },
  col: { flex: 1 },
  colLabel: { fontSize: 8, color: '#94a3b8' },
  colValue: { fontSize: 9, color: '#334155' },
  pill: {
    paddingHorizontal: 6, paddingVertical: 2, borderRadius: 99,
    fontSize: 8, fontFamily: 'Helvetica-Bold',
  },
  pillGreen: { backgroundColor: '#d1fae5', color: '#065f46' },
  pillRed:   { backgroundColor: '#fee2e2', color: '#991b1b' },
  narrative: { fontSize: 9, color: '#475569', marginTop: 2, fontStyle: 'italic' },
  barRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 4 },
  barLabel: { width: 130, fontSize: 8, color: '#475569' },
  barTrack: { flex: 1, height: 6, backgroundColor: '#e2e8f0', borderRadius: 3 },
  barFill:  { height: 6, backgroundColor: '#3b82f6', borderRadius: 3 },
  footer: { position: 'absolute', bottom: 30, left: 40, right: 40, fontSize: 8, color: '#94a3b8', textAlign: 'center' },
})

export function ReportDocument({ data, start, end }) {
  const shapEntries = Object.entries(data.shap_aggregate || {})
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
  const maxShap = Math.max(...shapEntries.map(([, v]) => v), 0.001)

  const compEntries = Object.entries(data.feature_comparison || {})
    .sort((a, b) => Math.abs(b[1].gap) - Math.abs(a[1].gap)).slice(0, 6)

  return (
    <Document title={`Rapport IA · ${start} → ${end}`}>
      <Page size="A4" style={S.page}>

        {/* Header */}
        <View style={S.header}>
          <Text style={S.title}>Rapport d'Analyse IA — CV-Intelligence</Text>
          <Text style={S.subtitle}>
            LuxTalent Advisory Group · Période : {start} → {end} · Modèle v3 Fairness-Aware
          </Text>
        </View>

        {/* KPIs */}
        <View style={S.kpiRow}>
          <View style={S.kpi}>
            <Text style={S.kpiLabel}>Candidats analysés</Text>
            <Text style={S.kpiValue}>{data.total}</Text>
          </View>
          <View style={[S.kpi, { borderLeftColor: '#10b981' }]}>
            <Text style={S.kpiLabel}>Invités</Text>
            <Text style={[S.kpiValue, { color: '#065f46' }]}>{data.invited}</Text>
          </View>
          <View style={[S.kpi, { borderLeftColor: '#ef4444' }]}>
            <Text style={S.kpiLabel}>Rejetés</Text>
            <Text style={[S.kpiValue, { color: '#991b1b' }]}>{data.rejected}</Text>
          </View>
          <View style={[S.kpi, { borderLeftColor: '#f59e0b' }]}>
            <Text style={S.kpiLabel}>Taux d'invitation</Text>
            <Text style={[S.kpiValue, { color: '#78350f' }]}>{data.invite_rate}%</Text>
          </View>
        </View>

        {/* SHAP */}
        {shapEntries.length > 0 && (
          <View style={S.section}>
            <Text style={S.sectionTitle}>Variables les plus influentes (SHAP agrégé)</Text>
            {shapEntries.map(([name, val]) => (
              <View key={name} style={S.barRow}>
                <Text style={S.barLabel}>{name}</Text>
                <View style={S.barTrack}>
                  <View style={[S.barFill, { width: `${val / maxShap * 100}%` }]} />
                </View>
                <Text style={{ fontSize: 8, color: '#64748b', width: 40, textAlign: 'right' }}>
                  {val.toFixed(4)}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Feature comparison */}
        {compEntries.length > 0 && (
          <View style={S.section}>
            <Text style={S.sectionTitle}>Profil moyen : Invités vs Rejetés</Text>
            <View style={[S.row, { borderBottom: '1.5px solid #cbd5e1' }]}>
              <Text style={[S.barLabel, { fontFamily: 'Helvetica-Bold', fontSize: 8 }]}>Variable</Text>
              <Text style={{ width: 70, fontSize: 8, fontFamily: 'Helvetica-Bold', color: '#065f46' }}>Moy. Invités</Text>
              <Text style={{ width: 70, fontSize: 8, fontFamily: 'Helvetica-Bold', color: '#991b1b' }}>Moy. Rejetés</Text>
              <Text style={{ width: 50, fontSize: 8, fontFamily: 'Helvetica-Bold', color: '#1e3a8a' }}>Écart</Text>
            </View>
            {compEntries.map(([name, v]) => (
              <View key={name} style={S.row}>
                <Text style={[S.barLabel, { fontSize: 9 }]}>{name}</Text>
                <Text style={{ width: 70, fontSize: 9, color: '#065f46' }}>{v.invited_avg}</Text>
                <Text style={{ width: 70, fontSize: 9, color: '#991b1b' }}>{v.rejected_avg}</Text>
                <Text style={{ width: 50, fontSize: 9, color: v.gap > 0 ? '#065f46' : '#991b1b' }}>
                  {v.gap > 0 ? '+' : ''}{v.gap}
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* Candidate list */}
        <View style={S.section}>
          <Text style={S.sectionTitle}>Liste des candidats ({data.candidates.length})</Text>
          <View style={[S.row, { borderBottom: '1.5px solid #cbd5e1' }]}>
            {['Nom', 'Secteur', 'Score', 'Décision', 'Explication IA'].map(h => (
              <Text key={h} style={{ flex: h === 'Explication IA' ? 3 : 1, fontSize: 8, fontFamily: 'Helvetica-Bold', color: '#64748b' }}>{h}</Text>
            ))}
          </View>
          {data.candidates.map((c, i) => (
            <View key={i} style={S.row} wrap={false}>
              <Text style={{ flex: 1, fontSize: 8 }}>{c.name || '—'}</Text>
              <Text style={{ flex: 1, fontSize: 8, color: '#64748b' }}>{c.sector || '—'}</Text>
              <Text style={{ flex: 1, fontSize: 8, fontFamily: 'Helvetica-Bold', color: parseFloat(c.score) >= 0.6 ? '#065f46' : '#991b1b' }}>
                {Math.round(parseFloat(c.score || 0) * 100)}%
              </Text>
              <Text style={{ flex: 1, fontSize: 8, color: c.decision === 'invite' ? '#065f46' : '#991b1b' }}>
                {c.decision === 'invite' ? 'Invité' : 'Rejeté'}
              </Text>
              <Text style={{ flex: 3, fontSize: 7, color: '#64748b', fontStyle: 'italic' }}>{c.narrative}</Text>
            </View>
          ))}
        </View>

        <Text style={S.footer}>
          Généré le {new Date().toLocaleDateString('fr-FR')} · CV-Intelligence v3 Fairness-Aware · LuxTalent Advisory Group
        </Text>
      </Page>
    </Document>
  )
}
