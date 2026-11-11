import { useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { ApiError, listRuns, listUsers } from "../lib/api"
import { useAuth } from "../lib/AuthContext"
import { ROLE_DESCRIPTIONS, ROLE_LABELS, canRunDetection } from "../lib/constants"
import type { Role, RunRow, UserRow } from "../lib/types"
import { formatDateTime } from "../lib/format"
import { StatusBadge } from "../components/StatusBadge"
import { EmptyState, ErrorBanner, PageHeading, Spinner } from "../components/States"

const HEADINGS: Record<Role, string> = {
  admin: "Administrator dashboard",
  operator: "Operator dashboard",
  analyst: "Analyst dashboard",
}

function StatCard({ label, value, testId }: { label: string; value: string | number; testId: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-4" data-testid={testId}>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-900">{value}</p>
    </div>
  )
}

function stats(role: Role, runs: RunRow[], mine: RunRow[], users: UserRow[] | null) {
  const active = (rows: RunRow[]) => rows.filter((r) => r.status === "queued" || r.status === "running").length
  const incidents = (rows: RunRow[]) => rows.reduce((sum, r) => sum + r.incident_count, 0)
  const done = runs.filter((r) => r.status === "done")
  if (role === "admin") {
    const activeUsers = users?.filter((u) => u.active) ?? null
    return [
      { label: "Runs", value: runs.length, testId: "stat-runs" },
      { label: "Incidents flagged", value: incidents(runs), testId: "stat-incidents" },
      { label: "In progress", value: active(runs), testId: "stat-active" },
      { label: "Active users", value: activeUsers === null ? "-" : activeUsers.length, testId: "stat-users" },
    ]
  }
  if (role === "operator") {
    return [
      { label: "My runs", value: mine.length, testId: "stat-my-runs" },
      { label: "Mine in progress", value: active(mine), testId: "stat-active" },
      { label: "Incidents in my runs", value: incidents(mine), testId: "stat-incidents" },
      { label: "All runs", value: runs.length, testId: "stat-runs" },
    ]
  }
  return [
    { label: "Completed runs", value: done.length, testId: "stat-done" },
    { label: "Incidents flagged", value: incidents(done), testId: "stat-incidents" },
    { label: "Runs with incidents", value: done.filter((r) => r.incident_count > 0).length, testId: "stat-with-incidents" },
    { label: "Latest run", value: done[0] ? formatDateTime(done[0].created_at) : "-", testId: "stat-latest" },
  ]
}

export default function Dashboard() {
  const { role, username } = useAuth()
  const [runs, setRuns] = useState<RunRow[] | null>(null)
  const [users, setUsers] = useState<UserRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [onlyMine, setOnlyMine] = useState(true)

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

  useEffect(() => {
    if (role !== "admin") return
    let cancelled = false
    listUsers()
      .then((rows) => {
        if (!cancelled) setUsers(rows)
      })
      .catch(() => {
        if (!cancelled) setUsers(null)
      })
    return () => {
      cancelled = true
    }
  }, [role])

  const mine = useMemo(() => (runs ?? []).filter((r) => r.created_by === username), [runs, username])

  if (role === null) return null

  const shown = role === "operator" && onlyMine ? mine : (runs ?? [])
  const canRun = canRunDetection(role)

  return (
    <div data-testid={`dashboard-${role}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <PageHeading title={HEADINGS[role]} description={`${ROLE_LABELS[role]}: ${ROLE_DESCRIPTIONS[role]}`} />
        <div className="flex gap-2">
          {role === "admin" && (
            <Link
              to="/users"
              className="rounded-md border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
            >
              Manage users
            </Link>
          )}
          {canRun && (
            <Link
              to="/upload"
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
            >
              New run
            </Link>
          )}
        </div>
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

      {runs !== null && (
        <>
          <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {stats(role, runs, mine, users).map((stat) => (
              <StatCard key={stat.testId} {...stat} />
            ))}
          </div>

          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900">
              {role === "operator" && onlyMine ? "My runs" : "All runs"}
            </h2>
            {role === "operator" && (
              <div className="flex rounded-md border border-slate-200 bg-white p-0.5 text-sm" role="group" aria-label="Which runs">
                {[
                  { mine: true, label: "Mine" },
                  { mine: false, label: "All" },
                ].map((option) => (
                  <button
                    key={option.label}
                    type="button"
                    aria-pressed={onlyMine === option.mine}
                    onClick={() => setOnlyMine(option.mine)}
                    className={`rounded px-3 py-1 font-medium ${
                      onlyMine === option.mine ? "bg-indigo-50 text-indigo-700" : "text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {shown.length === 0 && (
            <EmptyState
              title="No runs yet"
              description={
                canRun
                  ? "Upload a video to detect and track vehicles, and flag incidents."
                  : "Runs appear here once an operator processes a video."
              }
              action={
                canRun ? (
                  <Link
                    to="/upload"
                    className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500"
                  >
                    Upload a video
                  </Link>
                ) : undefined
              }
            />
          )}

          {shown.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-left text-sm" data-testid="runs-table">
                <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th scope="col" className="px-4 py-3 font-medium">Video</th>
                    <th scope="col" className="px-4 py-3 font-medium">Status</th>
                    <th scope="col" className="px-4 py-3 font-medium">Incidents</th>
                    <th scope="col" className="px-4 py-3 font-medium">Started by</th>
                    <th scope="col" className="px-4 py-3 font-medium">Created</th>
                    <th scope="col" className="px-4 py-3 font-medium">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {shown.map((run) => (
                    <tr key={run.id} className="hover:bg-slate-50">
                      <td className="max-w-xs truncate px-4 py-3 font-medium text-slate-900">
                        {run.video_filename}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={run.status} error={run.error} />
                      </td>
                      <td className="px-4 py-3 text-slate-600">{run.incident_count}</td>
                      <td className="px-4 py-3 text-slate-600">{run.created_by ?? "-"}</td>
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
        </>
      )}
    </div>
  )
}
