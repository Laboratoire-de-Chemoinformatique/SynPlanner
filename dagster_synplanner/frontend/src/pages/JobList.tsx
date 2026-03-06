import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api'

const statusColors: Record<string, string> = {
  SUCCESS: '#22c55e',
  FAILURE: '#ef4444',
  STARTED: '#3b82f6',
  QUEUED: '#a855f7',
  CANCELED: '#6b7280',
}

export default function JobList() {
  const { data: jobs, isLoading, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.listJobs(),
  })

  if (isLoading) return <p>Loading jobs...</p>
  if (error) return <p>Failed to load jobs. Is the API running?</p>

  return (
    <div>
      <h2>Recent Jobs</h2>
      {!jobs?.length ? (
        <p>No jobs yet. <Link to="/submit">Submit one</Link>.</p>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #ddd', textAlign: 'left' }}>
              <th style={{ padding: '0.5rem' }}>Run ID</th>
              <th style={{ padding: '0.5rem' }}>Job Type</th>
              <th style={{ padding: '0.5rem' }}>Status</th>
              <th style={{ padding: '0.5rem' }}>Started</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.run_id} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '0.5rem' }}>
                  <Link to={`/jobs/${job.run_id}`}>{job.run_id.slice(0, 8)}...</Link>
                </td>
                <td style={{ padding: '0.5rem' }}>{job.job_type}</td>
                <td style={{ padding: '0.5rem' }}>
                  <span style={{
                    color: statusColors[job.status] || '#333',
                    fontWeight: 'bold',
                  }}>
                    {job.status}
                  </span>
                </td>
                <td style={{ padding: '0.5rem' }}>
                  {job.started_at ? new Date(Number(job.started_at) * 1000).toLocaleString() : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
