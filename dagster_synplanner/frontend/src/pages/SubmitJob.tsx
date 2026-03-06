import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { api, SubmitJobRequest } from '../api'

type JobKind = 'planning' | 'data_preparation' | 'full_training_pipeline'

export default function SubmitJob() {
  const navigate = useNavigate()
  const [kind, setKind] = useState<JobKind>('planning')

  // Planning fields
  const [planningConfig, setPlanningConfig] = useState('')
  const [targets, setTargets] = useState('')
  const [rules, setRules] = useState('')
  const [blocks, setBlocks] = useState('')
  const [policy, setPolicy] = useState('')
  const [valueNet, setValueNet] = useState('')

  // Data prep fields
  const [stdConfig, setStdConfig] = useState('')
  const [filterConfig, setFilterConfig] = useState('')
  const [inputReactions, setInputReactions] = useState('')

  const submit = useMutation({
    mutationFn: (req: SubmitJobRequest) => api.submitJob(req),
    onSuccess: (data) => navigate(`/jobs/${data.run_id}`),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const req: SubmitJobRequest = { job_type: kind }

    if (kind === 'planning') {
      req.planning_config = {
        planning_config: planningConfig,
        targets,
        reaction_rules: rules,
        building_blocks: blocks,
        policy_network: policy,
        value_network: valueNet || undefined,
      }
    } else if (kind === 'data_preparation') {
      req.data_prep_config = {
        standardization_config: stdConfig,
        filtering_config: filterConfig,
        input_reactions: inputReactions,
      }
    }

    submit.mutate(req)
  }

  const inputStyle = { width: '100%', padding: '0.4rem', marginBottom: '0.5rem', boxSizing: 'border-box' as const }
  const labelStyle = { display: 'block', marginBottom: '0.2rem', fontWeight: 'bold' as const }

  return (
    <div>
      <h2>Submit Job</h2>
      <form onSubmit={handleSubmit} style={{ maxWidth: 600 }}>
        <div style={{ marginBottom: '1rem' }}>
          <label style={labelStyle}>Job Type</label>
          <select value={kind} onChange={(e) => setKind(e.target.value as JobKind)} style={inputStyle}>
            <option value="planning">Retrosynthetic Planning</option>
            <option value="data_preparation">Data Preparation</option>
            <option value="full_training_pipeline">Full Training Pipeline</option>
          </select>
        </div>

        {kind === 'planning' && (
          <>
            <div><label style={labelStyle}>Planning Config (YAML path)</label><input style={inputStyle} value={planningConfig} onChange={e => setPlanningConfig(e.target.value)} required /></div>
            <div><label style={labelStyle}>Targets File (.smi)</label><input style={inputStyle} value={targets} onChange={e => setTargets(e.target.value)} required /></div>
            <div><label style={labelStyle}>Reaction Rules</label><input style={inputStyle} value={rules} onChange={e => setRules(e.target.value)} required /></div>
            <div><label style={labelStyle}>Building Blocks</label><input style={inputStyle} value={blocks} onChange={e => setBlocks(e.target.value)} required /></div>
            <div><label style={labelStyle}>Policy Network</label><input style={inputStyle} value={policy} onChange={e => setPolicy(e.target.value)} required /></div>
            <div><label style={labelStyle}>Value Network (optional)</label><input style={inputStyle} value={valueNet} onChange={e => setValueNet(e.target.value)} /></div>
          </>
        )}

        {kind === 'data_preparation' && (
          <>
            <div><label style={labelStyle}>Standardization Config</label><input style={inputStyle} value={stdConfig} onChange={e => setStdConfig(e.target.value)} required /></div>
            <div><label style={labelStyle}>Filtering Config</label><input style={inputStyle} value={filterConfig} onChange={e => setFilterConfig(e.target.value)} required /></div>
            <div><label style={labelStyle}>Input Reactions File</label><input style={inputStyle} value={inputReactions} onChange={e => setInputReactions(e.target.value)} required /></div>
          </>
        )}

        {kind === 'full_training_pipeline' && (
          <p>Full training pipeline uses default configs from the data directory. Configure via Dagster UI for advanced options.</p>
        )}

        <button type="submit" disabled={submit.isPending}
          style={{ padding: '0.5rem 2rem', marginTop: '1rem', cursor: 'pointer' }}>
          {submit.isPending ? 'Submitting...' : 'Submit Job'}
        </button>

        {submit.isError && (
          <p style={{ color: '#ef4444' }}>Failed to submit: {String(submit.error)}</p>
        )}
      </form>
    </div>
  )
}
