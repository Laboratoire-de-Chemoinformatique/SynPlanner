import { Routes, Route, Link } from 'react-router-dom'
import JobList from './pages/JobList'
import SubmitJob from './pages/SubmitJob'
import JobDetail from './pages/JobDetail'

export default function App() {
  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '1rem', fontFamily: 'system-ui' }}>
      <header style={{ borderBottom: '2px solid #333', marginBottom: '1.5rem', paddingBottom: '0.5rem' }}>
        <h1 style={{ margin: 0 }}>SynPlanner</h1>
        <nav style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
          <Link to="/">Jobs</Link>
          <Link to="/submit">Submit Job</Link>
          <a href={`${window.location.protocol}//${window.location.hostname}:3000`} target="_blank" rel="noreferrer">
            Dagster UI
          </a>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<JobList />} />
        <Route path="/submit" element={<SubmitJob />} />
        <Route path="/jobs/:runId" element={<JobDetail />} />
      </Routes>
    </div>
  )
}
