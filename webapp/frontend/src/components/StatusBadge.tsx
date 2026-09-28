import { isCancelledError, isInterruptedError } from "../lib/format"
import type { RunStatus } from "../lib/types"

const STYLES: Record<RunStatus, string> = {
  queued: "bg-slate-100 text-slate-700 ring-slate-600/20",
  running: "bg-amber-100 text-amber-800 ring-amber-600/20",
  done: "bg-emerald-100 text-emerald-800 ring-emerald-600/20",
  failed: "bg-rose-100 text-rose-800 ring-rose-600/20",
}

const LABELS: Record<RunStatus, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
}

const NEUTRAL = "bg-slate-100 text-slate-700 ring-slate-600/20"

// A failed run whose error says the user cancelled it, or that a server restart cut it off, is
// shown as that rather than as a generic failure.
export function StatusBadge({ status, error }: { status: RunStatus; error?: string | null }) {
  const cancelled = status === "failed" && isCancelledError(error)
  const interrupted = status === "failed" && isInterruptedError(error)
  const label = cancelled ? "Cancelled" : interrupted ? "Interrupted" : LABELS[status]
  const style = cancelled || interrupted ? NEUTRAL : STYLES[status]

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${style}`}
    >
      {status === "running" && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-600" />
      )}
      {label}
    </span>
  )
}
