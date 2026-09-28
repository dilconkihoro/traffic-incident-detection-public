import { useEffect, useRef, useState } from "react"
import { Link, useParams } from "react-router-dom"
import {
  ApiError,
  fetchIncidentsCsv,
  fetchTracksCsv,
  getRun,
  incidentsCsvUrl,
  originalVideoUrl,
  tracksCsvUrl,
  videoUrl,
} from "../lib/api"
import type { IncidentCsvRow, RunRow, TrackCsvRow } from "../lib/types"
import { parseIncidentsCsv, parseTracksCsv } from "../lib/csv"
import { INCIDENT_TYPE_LABELS } from "../lib/constants"
import { formatDateTime, formatTrackIds, isCancelledError } from "../lib/format"
import { Stepper } from "../components/Stepper"
import { EmptyState, ErrorBanner, PageHeading, Spinner } from "../components/States"
import { StatusBadge } from "../components/StatusBadge"
import { VideoPanel } from "../components/VideoPanel"

export default function RunDetail() {
  const { id } = useParams<{ id: string }>()
  const [run, setRun] = useState<RunRow | null>(null)
  const [incidents, setIncidents] = useState<IncidentCsvRow[] | null>(null)
  const [tracks, setTracks] = useState<TrackCsvRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notFound, setNotFound] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  function seekToIncident(t: string) {
    const seconds = Number(t)
    const video = videoRef.current
    if (video === null || Number.isNaN(seconds)) return
    video.currentTime = seconds
    video.scrollIntoView({ behavior: "smooth", block: "center" })
    void video.play()
  }

  useEffect(() => {
    if (!id) return
    let cancelled = false

    getRun(id)
      .then((row) => {
        if (cancelled) return
        setRun(row)
        if (row.status !== "done") return
        return Promise.all([
          fetchIncidentsCsv(id).then((text) => {
            if (!cancelled) setIncidents(parseIncidentsCsv(text))
          }),
          fetchTracksCsv(id).then((text) => {
            if (!cancelled) setTracks(parseTracksCsv(text))
          }),
        ])
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) setNotFound(true)
        else setError(err instanceof ApiError ? err.message : "Failed to load this run.")
      })

    return () => {
      cancelled = true
    }
  }, [id])

  if (!id) return null

  if (notFound) {
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

  if (error) {
    return <ErrorBanner message={error} />
  }

  if (!run) {
    return (
      <div className="flex justify-center py-24">
        <Spinner className="h-6 w-6 text-indigo-600" />
      </div>
    )
  }

  if (run.status !== "done") {
    return (
      <div className="mx-auto max-w-2xl">
        <Stepper current="processing" />
        <PageHeading title={run.video_filename} />
        <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-4 py-4">
          <StatusBadge status={run.status} error={run.error} />
          <p className="text-sm text-slate-600">
            {run.status === "failed"
              ? isCancelledError(run.error)
                ? "This run was cancelled."
                : `This run failed: ${run.error ?? "unknown error"}.`
              : "This run has not finished yet."}
          </p>
        </div>
        {run.status !== "failed" && (
          <Link
            to={`/processing/${run.id}`}
            className="mt-4 inline-block text-sm font-medium text-indigo-600 hover:text-indigo-500"
          >
            View live progress
          </Link>
        )}
      </div>
    )
  }

  const distinctVehicles = tracks !== null ? new Set(tracks.map((t) => t.track_id)).size : null

  return (
    <div className="mx-auto max-w-4xl">
      <Stepper current="results" />
      <PageHeading
        title={run.video_filename}
        description={`Completed ${formatDateTime(run.created_at)} - ${run.incident_count} incident${run.incident_count === 1 ? "" : "s"} flagged`}
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <VideoPanel
          title="Before: original upload"
          src={originalVideoUrl(run.id)}
          errorMessage="The original upload is not available for this run, or the browser could not play it."
        />
        <VideoPanel
          title="After: detected and tracked"
          src={videoUrl(run.id)}
          videoRef={videoRef}
          errorMessage="This video could not be played in the browser. Use Download and open it in a video player."
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        The "after" video has a box drawn around every tracked vehicle, plus a red banner when an
        incident is flagged. Click an incident below to jump straight to that moment in it.
      </p>

      <div className="mt-8 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Incidents</h2>
          {incidents !== null && incidents.length > 0 && (
            <p className="mt-0.5 text-xs text-slate-500">
              In the order they were flagged. Click one to jump the video to that moment.
            </p>
          )}
        </div>
        <a
          href={incidentsCsvUrl(run.id)}
          download
          className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
        >
          Download CSV
        </a>
      </div>

      {incidents === null && (
        <div className="mt-3 flex justify-center py-12">
          <Spinner className="h-5 w-5 text-indigo-600" />
        </div>
      )}

      {incidents !== null && incidents.length === 0 && (
        <div className="mt-3">
          <EmptyState
            title="No incidents flagged"
            description="No rule fired. That is expected for free-flowing traffic. The three rules (abrupt stop, lane obstruction, collision-like) are built for fixed-camera road footage: a camera that does not move, looking down a road. Close-up, hand-held or crash-test footage is usually not picked up, even when something dramatic happens. The tracked-vehicles table below shows what the system did see."
          />
        </div>
      )}

      {incidents !== null && incidents.length > 0 && (
        <div className="mt-3 overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">ID</th>
                <th scope="col" className="px-4 py-3 font-medium">Type</th>
                <th scope="col" className="px-4 py-3 font-medium">Vehicle(s)</th>
                <th scope="col" className="px-4 py-3 font-medium">Onset</th>
                <th scope="col" className="px-4 py-3 font-medium">Flagged</th>
                <th scope="col" className="px-4 py-3 font-medium">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {incidents.map((incident) => (
                <tr
                  key={incident.incident_id}
                  role="button"
                  tabIndex={0}
                  onClick={() => seekToIncident(incident.t_flag_s)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault()
                      seekToIncident(incident.t_flag_s)
                    }
                  }}
                  className="cursor-pointer hover:bg-indigo-50 focus:bg-indigo-50 focus:outline-none"
                >
                  <td className="px-4 py-3 text-slate-500">{incident.incident_id}</td>
                  <td className="px-4 py-3 font-medium text-slate-900">
                    {INCIDENT_TYPE_LABELS[incident.incident_type] ?? incident.incident_type}
                  </td>
                  <td className="px-4 py-3 text-slate-600">
                    {formatTrackIds(incident.track_ids.split(";"))}
                  </td>
                  <td className="px-4 py-3 text-slate-600">{incident.t_onset_s}s</td>
                  <td className="px-4 py-3 text-slate-600">{incident.t_flag_s}s</td>
                  <td className="px-4 py-3 text-slate-600">{incident.score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-8 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Tracked vehicles</h2>
          <p className="mt-0.5 max-w-xl text-xs text-slate-500">
            Every vehicle's position in every frame: the raw data the incidents above and the
            annotated video are both built from. One row per vehicle per frame, so a busy video
            has many rows; this is reference detail, not something you need to read line by line.
          </p>
        </div>
        <a
          href={tracksCsvUrl(run.id)}
          download
          className="shrink-0 text-sm font-medium text-indigo-600 hover:text-indigo-500"
        >
          Download CSV
        </a>
      </div>

      {tracks === null && (
        <div className="mt-3 flex justify-center py-12">
          <Spinner className="h-5 w-5 text-indigo-600" />
        </div>
      )}

      {tracks !== null && tracks.length === 0 && (
        <div className="mt-3">
          <EmptyState
            title="No vehicles tracked"
            description="No vehicle was detected and tracked in this run."
          />
        </div>
      )}

      {tracks !== null && tracks.length > 0 && (
        <>
          <p className="mt-3 text-sm text-slate-600">
            <span className="font-semibold text-slate-900">{distinctVehicles}</span> distinct{" "}
            {distinctVehicles === 1 ? "vehicle" : "vehicles"} tracked across{" "}
            <span className="font-semibold text-slate-900">{tracks.length}</span> position records.
          </p>
          <div className="mt-3 max-h-96 overflow-y-auto overflow-x-auto rounded-xl border border-slate-200 bg-white">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Frame</th>
                  <th scope="col" className="px-4 py-3 font-medium">Time</th>
                  <th scope="col" className="px-4 py-3 font-medium">Vehicle</th>
                  <th scope="col" className="px-4 py-3 font-medium">Class</th>
                  <th scope="col" className="px-4 py-3 font-medium">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {tracks.map((track, index) => (
                  <tr key={`${track.track_id}-${track.frame_idx}-${index}`}>
                    <td className="px-4 py-3 text-slate-600">{track.frame_idx}</td>
                    <td className="px-4 py-3 text-slate-600">{track.t_s}s</td>
                    <td className="px-4 py-3 font-medium text-slate-900">#{track.track_id}</td>
                    <td className="px-4 py-3 text-slate-600">{track.cls_name}</td>
                    <td className="px-4 py-3 text-slate-500">
                      {track.x1}, {track.y1}, {track.x2}, {track.y2}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
