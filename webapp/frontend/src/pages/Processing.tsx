import { useEffect, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { ApiError, cancelRun, eventsUrl, getRun } from "../lib/api"
import type { FailedEventPayload, FrameEventPayload, IncidentEventPayload, RunRow } from "../lib/types"
import { INCIDENT_TYPE_LABELS } from "../lib/constants"
import { formatElapsedMs, formatSeconds, formatTrackIds, isCancelledError } from "../lib/format"
import { Stepper } from "../components/Stepper"
import { PageHeading, ErrorBanner, Spinner } from "../components/States"

type Phase = "loading" | "running" | "failed" | "notfound" | "error"

// The stream can miss a terminal event (or the run can be gone after a server restart), so the
// run's own status is checked on a timer as well, and a run that stops advancing is called out.
const POLL_MS = 3000
const STALL_MS = 90_000

export default function Processing() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<RunRow | null>(null)
  const [phase, setPhase] = useState<Phase>("loading")
  const [progress, setProgress] = useState<FrameEventPayload>({ done: 0, total: null })
  const [incidents, setIncidents] = useState<IncidentEventPayload[]>([])
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [lastAdvanceAt, setLastAdvanceAt] = useState(() => Date.now())
  const progressMarker = `${progress.done}:${run?.status ?? ""}`
  const seenMarker = useRef(progressMarker)

  // One clock drives both the elapsed display and the "no progress" check: whenever the frame
  // count or status differs from the last tick, that tick is when progress last advanced.
  useEffect(() => {
    if (phase !== "running") return
    const id = window.setInterval(() => {
      const tick = Date.now()
      setNow(tick)
      if (seenMarker.current !== progressMarker) {
        seenMarker.current = progressMarker
        setLastAdvanceAt(tick)
      }
    }, 1000)
    return () => window.clearInterval(id)
  }, [phase, progressMarker])

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let source: EventSource | null = null
    let poll: number | undefined

    const applyRow = (row: RunRow) => {
      setRun(row)
      setProgress((prev) =>
        row.frames_done > prev.done ? { done: row.frames_done, total: row.frames_total } : prev,
      )
      if (row.status === "done") {
        navigate(`/runs/${jobId}`, { replace: true })
      } else if (row.status === "failed") {
        setPhase("failed")
        setError(row.error ?? "the run failed")
      }
    }

    const openStream = () => {
      source = new EventSource(eventsUrl(jobId), { withCredentials: true })

      source.addEventListener("frame", (event) => {
        setProgress(JSON.parse((event as MessageEvent).data) as FrameEventPayload)
      })
      source.addEventListener("incident", (event) => {
        const incident = JSON.parse((event as MessageEvent).data) as IncidentEventPayload
        setIncidents((prev) => [incident, ...prev])
      })
      source.addEventListener("done", () => {
        source?.close()
        navigate(`/runs/${jobId}`, { replace: true })
      })
      source.addEventListener("failed", (event) => {
        const data = JSON.parse((event as MessageEvent).data) as FailedEventPayload
        source?.close()
        setPhase("failed")
        setError(data.error)
      })
      // No error handler needed: if the stream is unavailable (a 404 for a run nothing is
      // executing any more), the status poll below still reports the truth.
    }

    getRun(jobId)
      .then((row) => {
        if (cancelled) return
        applyRow(row)
        if (row.status === "queued" || row.status === "running") {
          setPhase("running")
          openStream()
          poll = window.setInterval(() => {
            getRun(jobId)
              .then((fresh) => {
                if (!cancelled) applyRow(fresh)
              })
              .catch(() => undefined)
          }, POLL_MS)
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) {
          setPhase("notfound")
        } else {
          setPhase("error")
          setError(err instanceof ApiError ? err.message : "Failed to load this run.")
        }
      })

    return () => {
      cancelled = true
      source?.close()
      window.clearInterval(poll)
    }
  }, [jobId, navigate])

  // Re-reads the run and moves the page to whatever the server says it now is.
  const syncRun = async (id: string): Promise<RunRow> => {
    const row = await getRun(id)
    setRun(row)
    if (row.status === "failed") {
      setPhase("failed")
      setError(row.error ?? "the run failed")
    } else if (row.status === "done") {
      navigate(`/runs/${id}`, { replace: true })
    }
    return row
  }

  const handleCancel = async () => {
    if (!jobId || cancelling) return
    if (!window.confirm("Cancel this run? Anything processed so far is discarded.")) return
    setActionError(null)
    setCancelling(true)
    try {
      await cancelRun(jobId)
      // A run with nothing executing it is cleared straight away; a live one stops within a
      // frame or two and reports through the stream. Either way the run's status says which.
      await syncRun(jobId)
    } catch (err) {
      setCancelling(false)
      const explanation =
        err instanceof ApiError && err.status === 404
          ? "The server could not cancel this run. If the backend was just updated, restart it and refresh this page."
          : err instanceof ApiError
            ? err.message
            : "Could not cancel this run."
      // The page may simply be out of date (the run already finished, failed, or the server was
      // restarted): if the server's view is final, show that instead of an error.
      try {
        const row = await syncRun(jobId)
        if (row.status === "queued" || row.status === "running") setActionError(explanation)
      } catch {
        setActionError(explanation)
      }
    }
  }

  if (!jobId) return null

  if (phase === "loading") {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="h-6 w-6 text-indigo-600" />
      </div>
    )
  }

  if (phase === "notfound") {
    return (
      <ErrorBanner
        message="This run could not be found."
        action={
          <Link to="/dashboard" className="text-sm font-medium text-indigo-600 hover:text-indigo-500">
            Back to dashboard
          </Link>
        }
      />
    )
  }

  if (phase === "error" || phase === "failed") {
    const message =
      phase === "failed"
        ? isCancelledError(error)
          ? "This run was cancelled."
          : `Run failed: ${error}`
        : (error ?? "Something went wrong.")
    return (
      <div className="mx-auto max-w-2xl">
        <Stepper current="processing" />
        <ErrorBanner message={message} />
        <div className="mt-6 flex gap-3">
          <Link
            to="/upload"
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500"
          >
            Start a new run
          </Link>
          <Link
            to="/dashboard"
            className="rounded-md border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
          >
            View dashboard
          </Link>
        </div>
      </div>
    )
  }

  const queued = run?.status === "queued"
  const percent =
    progress.total !== null ? Math.min(100, Math.round((progress.done / progress.total) * 100)) : null
  const elapsedMs = run ? now - new Date(run.created_at).getTime() : 0
  const stalled = !queued && now - lastAdvanceAt > STALL_MS

  return (
    <div className="mx-auto max-w-2xl">
      <Stepper current="processing" />
      <PageHeading title={queued ? "Waiting to start" : "Processing"} description={run?.video_filename} />

      <div className="rounded-xl border border-slate-200 bg-white p-6">
        {queued ? (
          <p className="text-sm text-slate-600">
            Another run is being processed. Only one runs at a time so this machine stays usable;
            this one starts as soon as that finishes.
          </p>
        ) : (
          <>
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-slate-700">
                {percent !== null
                  ? `${progress.done} / ${progress.total} frames`
                  : `${progress.done} frames processed`}
              </span>
              <span className="text-slate-400">Elapsed {formatElapsedMs(elapsedMs)}</span>
            </div>
            <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-slate-200">
              {percent !== null ? (
                <div
                  className="h-full rounded-full bg-indigo-600 transition-all duration-300"
                  style={{ width: `${percent}%` }}
                />
              ) : (
                <div className="h-full w-1/3 animate-pulse rounded-full bg-indigo-600" />
              )}
            </div>
            <p className="mt-3 text-xs text-slate-400">
              Running on CPU: this reflects real processing speed, not live 30 fps video.
            </p>
          </>
        )}

        {stalled && (
          <p className="mt-4 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
            No progress for {formatElapsedMs(now - lastAdvanceAt)}. If the server was
            restarted, this run is gone; cancel it to clear it.
          </p>
        )}
        {actionError && <p className="mt-4 text-sm text-rose-700">{actionError}</p>}

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={handleCancel}
            disabled={cancelling}
            className="rounded-md border border-rose-200 px-4 py-2 text-sm font-medium text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {cancelling ? "Cancelling..." : "Cancel run"}
          </button>
        </div>
      </div>

      <div className="mt-6">
        <h2 className="mb-3 text-sm font-semibold text-slate-900">
          Incidents flagged {incidents.length > 0 && `(${incidents.length})`}
        </h2>
        {incidents.length === 0 ? (
          <p className="rounded-lg border border-dashed border-slate-300 bg-white px-4 py-8 text-center text-sm text-slate-400">
            No incidents flagged yet.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {incidents.map((incident, index) => (
              <li
                key={`${incident.type}-${incident.frame_flag}-${index}`}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3"
              >
                <div>
                  <p className="text-sm font-medium text-slate-900">
                    {INCIDENT_TYPE_LABELS[incident.type] ?? incident.type}
                  </p>
                  <p className="text-xs text-slate-500">
                    at {formatSeconds(incident.t_flag_s)}, {formatTrackIds(incident.track_ids)}
                  </p>
                </div>
                <span className="text-xs font-medium text-slate-400">score {incident.score.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
