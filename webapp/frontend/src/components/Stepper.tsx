export type FlowStep = "upload" | "processing" | "results"

const STEPS: { key: FlowStep; label: string }[] = [
  { key: "upload", label: "Upload" },
  { key: "processing", label: "Processing" },
  { key: "results", label: "Results" },
]

export function Stepper({ current }: { current: FlowStep }) {
  const currentIndex = STEPS.findIndex((step) => step.key === current)

  return (
    <ol className="mb-8 flex items-center" aria-label="Run progress">
      {STEPS.map((step, index) => {
        const isCompleted = index < currentIndex
        const isActive = index === currentIndex
        return (
          <li key={step.key} className="flex flex-1 items-center last:flex-none">
            <div className="flex items-center gap-2.5">
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold ${
                  isCompleted
                    ? "bg-indigo-600 text-white"
                    : isActive
                      ? "bg-indigo-100 text-indigo-700 ring-2 ring-indigo-600"
                      : "bg-slate-100 text-slate-400"
                }`}
                aria-current={isActive ? "step" : undefined}
              >
                {isCompleted ? (
                  <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                    <path
                      fillRule="evenodd"
                      d="M16.7 5.3a1 1 0 0 1 0 1.4l-7.4 7.4a1 1 0 0 1-1.4 0L3.3 9.5a1 1 0 1 1 1.4-1.4l3.9 3.9 6.7-6.7a1 1 0 0 1 1.4 0Z"
                      clipRule="evenodd"
                    />
                  </svg>
                ) : (
                  index + 1
                )}
              </span>
              <span
                className={`text-sm font-medium ${isActive || isCompleted ? "text-slate-900" : "text-slate-400"}`}
              >
                {step.label}
              </span>
            </div>
            {index < STEPS.length - 1 && (
              <span
                className={`mx-4 h-px flex-1 ${isCompleted ? "bg-indigo-600" : "bg-slate-200"}`}
              />
            )}
          </li>
        )
      })}
    </ol>
  )
}
