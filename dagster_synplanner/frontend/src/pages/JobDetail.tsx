import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function JobDetail() {
  const { runId } = useParams<{ runId: string }>()

  const { data: job, isLoading } = useQuery({
    queryKey: ['job', runId],
    queryFn: () => api.getJob(runId!),
    enabled: !!runId,
  })

  const { data: results } = useQuery({
    queryKey: ['results', runId],
    queryFn: () => api.listResults(runId!),
    enabled: !!runId && job?.status === 'SUCCESS',
  })

  if (isLoading) return <p>Loading...</p>
  if (!job) return <p>Job not found</p>

  return (
    <div>
      <h2>Job Details</h2>
      <table style={{ borderCollapse: 'collapse' }}>
        <tbody>
          <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Run ID</td><td>{job.run_id}</td></tr>
          <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Type</td><td>{job.job_type}</td></tr>
          <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Status</td><td>{job.status}</td></tr>
          <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Started</td><td>{job.started_at ? new Date(Number(job.started_at) * 1000).toLocaleString() : '-'}</td></tr>
          <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Ended</td><td>{job.ended_at ? new Date(Number(job.ended_at) * 1000).toLocaleString() : '-'}</td></tr>
          {job.error_message && (
            <tr><td style={{ padding: '0.3rem 1rem', fontWeight: 'bold' }}>Error</td><td style={{ color: '#ef4444' }}>{job.error_message}</td></tr>
          )}
        </tbody>
      </table>

      {job.status === 'STARTED' && (
        <p style={{ marginTop: '1rem' }}>
          Job is running. This page auto-refreshes every 5 seconds.
          <br />
          <a href={`${window.location.protocol}//${window.location.hostname}:3000/runs/${job.run_id}`} target="_blank" rel="noreferrer">
            View in Dagster UI
          </a>
        </p>
      )}

      {results?.files && results.files.length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <h3>Results</h3>
          <ul>
            {results.files.map((f) => (
              <li key={f.path}>
                <a href={`/api/results/${runId}/download/${f.path}`}>{f.name}</a>
                {' '}({(f.size / 1024).toFixed(1)} KB)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
