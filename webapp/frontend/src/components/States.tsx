import type { ReactNode } from "react"

export function Spinner({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg className={`animate-spin text-current ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8V0C5.4 0 0 5.4 0 12h4Z"
      />
    </svg>
  )
}

export function FullPageSpinner() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50">
      <Spinner className="h-8 w-8 text-indigo-600" />
    </div>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      {description && <p className="mt-1.5 max-w-sm text-sm text-slate-500">{description}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}

export function ErrorBanner({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-start gap-2.5">
        <svg
          viewBox="0 0 20 20"
          fill="currentColor"
          className="mt-0.5 h-5 w-5 shrink-0 text-rose-500"
        >
          <path
            fillRule="evenodd"
            d="M8.5 3.6a1.7 1.7 0 0 1 3 0l6.4 11.2c.6 1.1-.2 2.5-1.5 2.5H3.6c-1.3 0-2.1-1.4-1.5-2.5L8.5 3.6ZM10 7a.9.9 0 0 1 .9.9v3.4a.9.9 0 1 1-1.8 0V7.9A.9.9 0 0 1 10 7Zm0 7.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"
            clipRule="evenodd"
          />
        </svg>
        <p className="text-sm font-medium text-rose-800">{message}</p>
      </div>
      {action}
    </div>
  )
}

export function PageHeading({ title, description }: { title: string; description?: string }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
      {description && <p className="mt-1 text-sm text-slate-500">{description}</p>}
    </div>
  )
}
