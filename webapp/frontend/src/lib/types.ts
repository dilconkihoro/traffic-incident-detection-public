// Shapes mirror the backend exactly as observed in webapp/backend (db.py, routers/*.py,
// trafficwatch/types.py, src/trafficwatch/incident_log.py). Never widen these speculatively;
// update them if the backend contract changes.

export type RunStatus = "queued" | "running" | "done" | "failed"

export interface RunRow {
  id: string
  video_filename: string
  created_at: string
  status: RunStatus
  run_id: string | null
  incident_count: number
  frames_total: number | null
  frames_done: number
  error: string | null
  upload_path: string | null
}

export interface LoginResponse {
  username: string
}

export interface UploadResponse {
  upload_id: string
  path: string
  filename: string
}

export interface CreateRunPayload {
  upload_id: string
  path: string
  filename: string
  standin: boolean
}

export interface CreateRunResponse {
  job_id: string
}

export type IncidentType = "ABRUPT_STOP" | "LANE_OBSTRUCTION" | "COLLISION_LIKE"

export interface IncidentEventPayload {
  type: IncidentType
  track_ids: number[]
  frame_onset: number
  frame_flag: number
  t_onset_s: number
  t_flag_s: number
  bbox_union: [number, number, number, number]
  score: number
}

export interface FrameEventPayload {
  done: number
  total: number | null
}

export interface DoneEventPayload {
  run_id: string
}

export interface FailedEventPayload {
  error: string
}

// One row of reports/<run_id>/incidents.csv (src/trafficwatch/incident_log.py INCIDENT_COLUMNS).
export interface IncidentCsvRow {
  incident_id: string
  run_id: string
  video_id: string
  incident_type: string
  track_ids: string
  frame_onset: string
  frame_flag: string
  t_onset_s: string
  t_flag_s: string
  x1: string
  y1: string
  x2: string
  y2: string
  score: string
  rule_params_hash: string
  weights_run_id: string
  weights_kind: string
}

// One row of reports/<run_id>/tracks.csv (src/trafficwatch/incident_log.py TRACK_COLUMNS): one
// tracked vehicle's position in one frame. This is the raw trajectory data the incident rules
// and the annotated video are both built from.
export interface TrackCsvRow {
  video_id: string
  frame_idx: string
  t_s: string
  track_id: string
  cls_name: string
  x1: string
  y1: string
  x2: string
  y2: string
}
