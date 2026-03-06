import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
})

export interface JobStatus {
  run_id: string
  job_type: string
  status: string
  started_at: string | null
  ended_at: string | null
  error_message: string | null
}

export interface SubmitJobRequest {
  job_type: string
  planning_config?: {
    planning_config: string
    targets: string
    reaction_rules: string
    building_blocks: string
    policy_network: string
    value_network?: string
    cluster_results?: boolean
  }
  data_prep_config?: {
    standardization_config: string
    filtering_config: string
    input_reactions: string
    num_cpus?: number
    batch_size?: number
  }
  training_config?: {
    policy_config: string
    rule_extraction_config: string
    input_reactions: string
    num_cpus?: number
  }
}

export const api = {
  listJobs: (limit = 20) =>
    client.get<JobStatus[]>('/jobs', { params: { limit } }).then((r) => r.data),

  getJob: (runId: string) =>
    client.get<JobStatus>(`/jobs/${runId}`).then((r) => r.data),

  submitJob: (req: SubmitJobRequest) =>
    client.post<{ run_id: string; status: string }>('/jobs/submit', req).then((r) => r.data),

  uploadFile: (category: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return client
      .post<{ path: string; size: number }>(`/upload/${category}`, form)
      .then((r) => r.data)
  },

  listResults: (runId: string) =>
    client.get<{ run_id: string; files: { name: string; path: string; size: number }[] }>(
      `/results/${runId}`
    ).then((r) => r.data),

  health: () => client.get('/health').then((r) => r.data),
}
