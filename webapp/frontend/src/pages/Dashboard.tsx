import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { ApiError, listRuns } from "../lib/api"
import type { RunRow } from "../lib/types"
import { formatDateTime } from "../lib/format"
import { StatusBadge } from "../components/StatusBadge"
import { EmptyState, ErrorBanner, PageHeading, Spinner } from "../components/States"

export default function Dashboard() {
  const [runs, setRuns] = useState<RunRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const load = () => {
      listRuns()
        .then((rows) => {
          if (!cancelled) {
            setRuns(rows)
            setError(null)
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) setError(err instanceof ApiError ? err.message : "Failed to load runs.")
        })
    }

    load()
    // Runs still in progress can change status between polls; a short interval keeps the list
    // fresh without needing a stream on a page that shows many runs at once.
    const interval = window.setInterval(load, 4000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeading title="Runs" description="Every video processed through the dashboard, newest first." />
        <Link
          to="/upload"
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
        >
          New run
        </Link>
      </div>

      {error && (
        <div className="mb-4">
          <ErrorBanner message={error} />
        </div>
      )}

      {runs === null && !error && (
        <div className="flex justify-center py-16">
          <Spinner className="h-6 w-6 text-indigo-600" />
        </div>
      )}

      {runs !== null && runs.length === 0 && (
        <EmptyState
          title="No runs yet"
          description="Upload a video to detect and track vehicles, and flag incidents."
          action={
            <Link
              to="/upload"
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500"
            >
              Upload a video
            </Link>
          }
        />
      )}

      {runs !== null && runs.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Video</th>
                <th scope="col" className="px-4 py-3 font-medium">Status</th>
                <th scope="col" className="px-4 py-3 font-medium">Incidents</th>
                <th scope="col" className="px-4 py-3 font-medium">Created</th>
                <th scope="col" className="px-4 py-3 font-medium">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((run) => (
                <tr key={run.id} className="hover:bg-slate-50">
                  <td className="max-w-xs truncate px-4 py-3 font-medium text-slate-900">
                    {run.video_filename}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={run.status} error={run.error} />
                  </td>
                  <td className="px-4 py-3 text-slate-600">{run.incident_count}</td>
                  <td className="px-4 py-3 text-slate-500">{formatDateTime(run.created_at)}</td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      to={run.status === "done" ? `/runs/${run.id}` : `/processing/${run.id}`}
                      className="font-medium text-indigo-600 hover:text-indigo-500"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
