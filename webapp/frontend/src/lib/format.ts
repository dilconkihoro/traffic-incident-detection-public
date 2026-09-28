export function formatSeconds(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${seconds.toString().padStart(2, "0")}`
}

export function formatElapsedMs(ms: number): string {
  return formatSeconds(ms / 1000)
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  })
}

// ABRUPT_STOP and LANE_OBSTRUCTION always report one vehicle; COLLISION_LIKE always reports two
// (src/trafficwatch/behaviour/rules.py). Spelling that out avoids reading "2" as "2 vehicles"
// when it is one vehicle whose id happens to be 2.
export function formatTrackIds(ids: readonly (string | number)[]): string {
  const labelled = ids.map((id) => `#${id}`)
  if (labelled.length <= 1) return `Vehicle ${labelled[0] ?? ""}`
  return `Vehicles ${labelled.slice(0, -1).join(", ")} and ${labelled[labelled.length - 1]}`
}

// The backend records a user-cancelled run as failed with an error starting "cancelled" (and one
// cut off by a server restart with "interrupted"); the UI shows those as what they are.
export function isCancelledError(error: string | null | undefined): boolean {
  return typeof error === "string" && error.startsWith("cancelled")
}

export function isInterruptedError(error: string | null | undefined): boolean {
  return typeof error === "string" && error.startsWith("interrupted")
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ["KB", "MB", "GB", "TB"]
  let value = bytes
  let unitIndex = -1
  do {
    value /= 1024
    unitIndex += 1
  } while (value >= 1024 && unitIndex < units.length - 1)
  return `${value.toFixed(1)} ${units[unitIndex]}`
}
