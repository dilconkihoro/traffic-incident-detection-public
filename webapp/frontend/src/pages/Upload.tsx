import { useRef, useState, type DragEvent } from "react"
import { useNavigate } from "react-router-dom"
import { createRun, uploadVideo, ApiError } from "../lib/api"
import { ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_BYTES, MAX_UPLOAD_DURATION_S } from "../lib/constants"
import { formatBytes } from "../lib/format"
import { Stepper } from "../components/Stepper"
import { PageHeading, ErrorBanner, Spinner } from "../components/States"

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".")
  return dot === -1 ? "" : filename.slice(dot).toLowerCase()
}

function validate(file: File): string | null {
  const extension = extensionOf(file.name)
  if (!ALLOWED_VIDEO_EXTENSIONS.includes(extension as (typeof ALLOWED_VIDEO_EXTENSIONS)[number])) {
    return `Unsupported file type "${extension || "unknown"}". Allowed: ${ALLOWED_VIDEO_EXTENSIONS.join(", ")}.`
  }
  if (file.size === 0) {
    return "The selected file is empty."
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File exceeds the ${formatBytes(MAX_UPLOAD_BYTES)} limit.`
  }
  return null
}

// Reads a video file's duration from its metadata only (no full download/decode needed), so this
// is fast even for a large file. Used purely as an early, upload-avoiding heads-up; the backend's
// exact frame count is what actually gets enforced.
function readDurationSeconds(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video")
    video.preload = "metadata"
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src)
      resolve(video.duration)
    }
    video.onerror = () => {
      URL.revokeObjectURL(video.src)
      reject(new Error("could not read video metadata"))
    }
    video.src = URL.createObjectURL(file)
  })
}

export default function Upload() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [duration, setDuration] = useState<number | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [stage, setStage] = useState<"idle" | "uploading" | "starting">("idle")

  const busy = stage !== "idle"

  const pickFile = async (candidate: File) => {
    setSubmitError(null)
    const problem = validate(candidate)
    if (problem) {
      setValidationError(problem)
      setFile(null)
      setDuration(null)
      return
    }

    try {
      const seconds = await readDurationSeconds(candidate)
      if (seconds > MAX_UPLOAD_DURATION_S) {
        setValidationError(
          `This clip is about ${Math.round(seconds)}s long. Keep uploads under ${MAX_UPLOAD_DURATION_S}s so processing stays fast on this machine (CPU only, no GPU).`,
        )
        setFile(null)
        setDuration(null)
        return
      }
      setValidationError(null)
      setDuration(seconds)
      setFile(candidate)
    } catch {
      // Metadata could not be read (an unusual container, say); let the server be the judge
      // rather than blocking the user on a check that only ever helps, never gatekeeps for real.
      setValidationError(null)
      setDuration(null)
      setFile(candidate)
    }
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    const dropped = event.dataTransfer.files[0]
    if (dropped) void pickFile(dropped)
  }

  const handleSubmit = async () => {
    if (!file) return
    setSubmitError(null)
    try {
      setStage("uploading")
      setProgress(0)
      const upload = await uploadVideo(file, setProgress)

      setStage("starting")
      const run = await createRun({
        upload_id: upload.upload_id,
        path: upload.path,
        filename: upload.filename,
        standin: true,
      })

      navigate(`/processing/${run.job_id}`)
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Upload failed. Check your connection and try again."
      setSubmitError(message)
      setStage("idle")
      setProgress(null)
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <Stepper current="upload" />
      <PageHeading
        title="Upload a CCTV clip"
        description="Pre-recorded fixed-camera footage only. Processing runs on CPU, so expect roughly real time or slower, not live 30 fps analysis."
      />

      <div
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors ${
          isDragging ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white"
        }`}
      >
        <svg viewBox="0 0 24 24" fill="none" className="h-10 w-10 text-slate-400">
          <path
            d="M12 16V4m0 0 4 4m-4-4-4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="mt-3 text-sm font-medium text-slate-700">
          Drag and drop a video, or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="text-indigo-600 hover:text-indigo-500"
          >
            browse your files
          </button>
        </p>
        <p className="mt-1 text-xs text-slate-400">
          {ALLOWED_VIDEO_EXTENSIONS.join(", ")}, up to {formatBytes(MAX_UPLOAD_BYTES)}, under{" "}
          {MAX_UPLOAD_DURATION_S}s (CPU-only processing, so short clips stay fast)
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_VIDEO_EXTENSIONS.join(",")}
          className="hidden"
          onChange={(event) => {
            const selected = event.target.files?.[0]
            if (selected) void pickFile(selected)
          }}
        />
      </div>

      {validationError && (
        <div className="mt-4">
          <ErrorBanner message={validationError} />
        </div>
      )}

      {file && !validationError && (
        <div className="mt-4 flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-slate-900">{file.name}</p>
            <p className="text-xs text-slate-500">
              {formatBytes(file.size)}
              {duration !== null && ` - ${duration.toFixed(1)}s`}
            </p>
          </div>
          {!busy && (
            <button
              type="button"
              onClick={() => {
                setFile(null)
                setDuration(null)
              }}
              className="ml-4 shrink-0 text-sm text-slate-400 hover:text-slate-600"
            >
              Remove
            </button>
          )}
        </div>
      )}

      {submitError && (
        <div className="mt-4">
          <ErrorBanner message={submitError} />
        </div>
      )}

      {stage === "uploading" && progress !== null && (
        <div className="mt-4">
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-indigo-600 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-1.5 text-xs text-slate-500">Uploading... {progress}%</p>
        </div>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={!file || busy}
        className="mt-6 flex w-full items-center justify-center gap-2 rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy && <Spinner className="h-4 w-4" />}
        {stage === "uploading" ? "Uploading..." : stage === "starting" ? "Starting run..." : "Start processing"}
      </button>
    </div>
  )
}
