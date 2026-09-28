import { useState, type RefObject } from "react"

interface VideoPanelProps {
  title: string
  src: string
  errorMessage: string
  videoRef?: RefObject<HTMLVideoElement | null>
}

export function VideoPanel({ title, src, errorMessage, videoRef }: VideoPanelProps) {
  const [failed, setFailed] = useState(false)

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
        <a href={src} download className="text-xs font-medium text-indigo-600 hover:text-indigo-500">
          Download
        </a>
      </div>
      {failed ? (
        <div className="flex aspect-video items-center justify-center rounded-xl border border-slate-200 bg-slate-50 px-6 text-center text-sm text-slate-500">
          {errorMessage}
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-black">
          <video
            ref={videoRef}
            controls
            className="aspect-video w-full"
            src={src}
            onError={() => setFailed(true)}
          >
            Your browser does not support embedded video playback.
          </video>
        </div>
      )}
    </div>
  )
}
