import { useEffect, useMemo, useState } from 'react'
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const MODES = {
  full: 'Full activation',
  heuristic_historical: 'Heuristic (historical)',
  heuristic_recentQ: 'Heuristic (recent quarter)',
}

const percent = value => `${(100 * Number(value || 0)).toFixed(1)}%`
const number = (value, digits = 1) => Number(value || 0).toFixed(digits)

export default function EVSimulatorApp() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [station, setStation] = useState('')
  const [mode, setMode] = useState('full')
  const [faults, setFaults] = useState('ON')
  const [fleet, setFleet] = useState(2)

  useEffect(() => {
    fetch('/simulator-data.json')
      .then(response => {
        if (!response.ok) throw new Error('Run Code/export_simulator_data.py after Week 9.')
        return response.json()
      })
      .then(payload => {
        setData(payload)
        const firstStation = payload.pareto?.[0]?.station || ''
        setStation(firstStation)
      })
      .catch(reason => setError(reason.message))
  }, [])

  const rows = useMemo(() => (data?.pareto || []).filter(row =>
    row.station === station && row.activation_mode === mode && row.faults === faults
  ), [data, station, mode, faults])

  const fleets = useMemo(() => [...new Set(rows.map(row => row.n_chargers))]
    .sort((left, right) => left - right), [rows])

  const activeFleet = fleets.includes(fleet) ? fleet : (fleets[0] ?? fleet)
  const selected = rows.find(row => row.n_chargers === activeFleet)
  const simulationDays = Number(data?.simulation_days || 30)
  const stations = [...new Set((data?.pareto || []).map(row => row.station))]
  const scheduleScenario = mode === 'heuristic_recentQ' ? 'recent_quarter' : 'historical_avg'
  const schedule = (data?.schedules || []).filter(row =>
    row.station === station && (scheduleScenario === 'historical_avg'
      ? row.demand_scenario === 'historical_avg'
      : row.demand_scenario.startsWith('recent_quarter'))
  ).map(row => ({
    hour: row.hour,
    arrivals: row.lambda_h,
    active: mode === 'full' ? activeFleet : Math.min(activeFleet, row.s_heuristic),
  }))

  if (error) return <main className="sim-message"><h2>Simulator data is not ready</h2><p>{error}</p></main>
  if (!data || !station) return <main className="sim-message">Loading simulator data…</main>

  const cards = selected ? [
    ['Realized utilization', percent(selected.mean_utilization_mean)],
    ['Static utilization', percent(selected.mean_utilization_static_mean)],
    ['P(wait > 15 min)', percent(selected.p_wait_gt_15min_mean)],
    ['Mean wait', `${number(selected.mean_wait_mean)} min`],
    ['Active charger-hours/day', number(selected.active_charger_hours_mean / simulationDays)],
    ['Scheduled charger-hours/day', number(selected.scheduled_charger_hours_mean / simulationDays)],
    ['Sessions/day', number(selected.n_sessions_mean / simulationDays)],
    ['Successful throughput/hour', number(selected.throughput_per_hour_mean, 2)],
    ['Customers faulted', percent(selected.fault_fraction_mean)],
  ] : []

  return <main className="simulator">
    <header>
      <h1>EV Charging Queue Simulator</h1>
      <p>Exact, precomputed simulation points. Odd fleet sizes are intentionally not interpolated.</p>
    </header>

    <section className="controls">
      <label>Station<select value={station} onChange={event => setStation(event.target.value)}>
        {stations.map(value => <option key={value}>{value}</option>)}
      </select></label>
      <label>Activation<select value={mode} onChange={event => setMode(event.target.value)}>
        {Object.entries(MODES).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select></label>
      <label>Faults<select value={faults} onChange={event => setFaults(event.target.value)}>
        <option>ON</option><option>OFF</option>
      </select></label>
      <label>Fleet size<select value={activeFleet} onChange={event => setFleet(Number(event.target.value))}>
        {fleets.map(value => <option key={value} value={value}>{value}</option>)}
      </select></label>
    </section>

    {!selected ? <section className="sim-message">This combination was not simulated.</section> : <>
      <section className="metric-grid">
        {cards.map(([label, value]) => <article key={label}><span>{label}</span><strong>{value}</strong></article>)}
      </section>

      <section className="chart-card">
        <h2>Service level frontier</h2>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={rows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="n_chargers" label={{ value: 'Fleet size', position: 'bottom' }} />
            <YAxis tickFormatter={value => `${Math.round(value * 100)}%`} />
            <Tooltip formatter={value => percent(value)} />
            <Legend />
            <Line dataKey="p_wait_gt_15min_mean" name="P(wait > 15 min)" stroke="#dc2626" />
            <Line dataKey="mean_utilization_mean" name="Realized utilization" stroke="#2563eb" />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {schedule.length > 0 && <section className="chart-card">
        <h2>Hourly schedule</h2>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={schedule}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" /><YAxis /><Tooltip /><Legend />
            <Bar dataKey="active" name="Active chargers" fill="#3b82f6" />
            <Line dataKey="arrivals" name="Arrival rate/hour" stroke="#f97316" />
          </ComposedChart>
        </ResponsiveContainer>
      </section>}
    </>}
  </main>
}
