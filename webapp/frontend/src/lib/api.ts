import type {
  CreateRunPayload,
  CreateRunResponse,
  LoginResponse,
  RunRow,
  UploadResponse,
} from "./types"

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// Subscriber the AuthContext installs so any 401, from any call anywhere in the app, can flip
// global auth state and trigger a redirect, without every call site handling it by hand.
type UnauthorizedHandler = () => void
let unauthorizedHandler: UnauthorizedHandler | null = null

export function setUnauthorizedHandler(handler: UnauthorizedHandler | null): void {
  unauthorizedHandler = handler
}

async function readDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.clone().json()
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail?: unknown }).detail
      if (typeof detail === "string") return detail
    }
  } catch {
    // Body wasn't JSON (or wasn't readable); fall back to a generic status message.
  }
  return null
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const hasJsonBody = init?.body !== undefined && !(init.body instanceof FormData)
  const response = await fetch(path, {
    ...init,
    credentials: "include",
    headers: {
      ...(hasJsonBody ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  })

  if (response.status === 401) {
    unauthorizedHandler?.()
    throw new ApiError(401, (await readDetail(response)) ?? "not authenticated")
  }
  if (!response.ok) {
    const detail = await readDetail(response)
    throw new ApiError(response.status, detail ?? `request failed with status ${response.status}`)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export function login(username: string, password: string): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/api/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  })
}

export function logout(): Promise<{ ok: boolean }> {
  return apiFetch<{ ok: boolean }>("/api/logout", { method: "POST" })
}

export function listRuns(): Promise<RunRow[]> {
  return apiFetch<RunRow[]>("/api/runs")
}

export function getRun(id: string): Promise<RunRow> {
  return apiFetch<RunRow>(`/api/runs/${encodeURIComponent(id)}`)
}

export function createRun(payload: CreateRunPayload): Promise<CreateRunResponse> {
  return apiFetch<CreateRunResponse>("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export function cancelRun(id: string): Promise<{ status: string }> {
  return apiFetch<{ status: string }>(`/api/runs/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  })
}

export function videoUrl(id: string): string {
  return `/api/runs/${encodeURIComponent(id)}/video`
}

export function originalVideoUrl(id: string): string {
  return `/api/runs/${encodeURIComponent(id)}/original`
}

export function incidentsCsvUrl(id: string): string {
  return `/api/runs/${encodeURIComponent(id)}/incidents.csv`
}

export function tracksCsvUrl(id: string): string {
  return `/api/runs/${encodeURIComponent(id)}/tracks.csv`
}

export function eventsUrl(id: string): string {
  return `/api/runs/${encodeURIComponent(id)}/events`
}

async function fetchCsv(url: string, label: string): Promise<string> {
  const response = await fetch(url, { credentials: "include" })
  if (response.status === 401) {
    unauthorizedHandler?.()
    throw new ApiError(401, "not authenticated")
  }
  if (!response.ok) {
    throw new ApiError(response.status, `failed to fetch ${label} (status ${response.status})`)
  }
  return response.text()
}

export function fetchIncidentsCsv(id: string): Promise<string> {
  return fetchCsv(incidentsCsvUrl(id), "incidents")
}

export function fetchTracksCsv(id: string): Promise<string> {
  return fetchCsv(tracksCsvUrl(id), "tracks")
}

export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/uploads")
    xhr.withCredentials = true

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status === 401) {
        unauthorizedHandler?.()
        reject(new ApiError(401, "not authenticated"))
        return
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as UploadResponse)
        return
      }
      let detail = `upload failed with status ${xhr.status}`
      try {
        const parsed: unknown = JSON.parse(xhr.responseText)
        if (parsed && typeof parsed === "object" && "detail" in parsed) {
          const value = (parsed as { detail?: unknown }).detail
          if (typeof value === "string") detail = value
        }
      } catch {
        // Non-JSON error body; keep the generic status message.
      }
      reject(new ApiError(xhr.status, detail))
    }

    xhr.onerror = () => reject(new ApiError(0, "network error during upload"))

    const formData = new FormData()
    formData.append("file", file)
    xhr.send(formData)
  })
}
