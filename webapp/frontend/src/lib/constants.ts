import type { Role } from "./types"

// Mirrors webapp/backend/routers/uploads.py exactly. Keep in sync if the backend limits change;
// this is a client-side sanity check only, the backend remains the source of truth and enforces
// the same limits itself.
export const ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"] as const

export const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

// The backend's real limit is a frame count (TW_MAX_UPLOAD_FRAMES, default 180), enforced after
// upload since only the server can decode the file to count frames exactly. This duration bound
// is a conservative, upload-avoiding pre-check: even at a low 10 fps, 30s is already ~300 frames,
// well past the default limit, so anything longer than this will fail either way. Shorter clips
// that still exceed the real frame limit (e.g. a high-fps clip) are caught by the server instead.
export const MAX_UPLOAD_DURATION_S = 30

export const INCIDENT_TYPE_LABELS: Record<string, string> = {
  ABRUPT_STOP: "Abrupt stop",
  LANE_OBSTRUCTION: "Lane obstruction",
  COLLISION_LIKE: "Collision-like convergence",
}

// Roles mirror webapp/backend/db.py ROLES and the access rules enforced by the backend
// (webapp/backend/main.py and routers/*.py). The UI hides what a role cannot do; the backend
// refuses it regardless.
export const ROLE_LABELS: Record<Role, string> = {
  admin: "Administrator",
  operator: "Traffic operator",
  analyst: "Analyst",
}

export const ROLE_DESCRIPTIONS: Record<Role, string> = {
  admin: "Manages user accounts and roles, and can do everything an operator and analyst can.",
  operator: "Loads CCTV videos, runs detection, cancels their own runs and reviews the results.",
  analyst: "Read-only: reviews runs, incidents and vehicles, and downloads the alert logs.",
}

export function canRunDetection(role: Role | null): boolean {
  return role === "admin" || role === "operator"
}

export function canCancelRun(role: Role | null, username: string | null, createdBy: string | null): boolean {
  return role === "admin" || (role === "operator" && createdBy !== null && createdBy === username)
}
