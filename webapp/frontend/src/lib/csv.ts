import type { IncidentCsvRow, TrackCsvRow } from "./types"

// Every field either producer writes (src/trafficwatch/incident_log.py) is a plain number, an
// enum value, or ids joined with ";" specifically to avoid colliding with the column delimiter,
// so a plain split is safe here; a quoted-field CSV parser would be dead weight for this fixed
// format.
function parseCsv<T>(text: string): T[] {
  const lines = text.split("\n").filter((line) => line.length > 0)
  if (lines.length === 0) return []

  const header = lines[0].split(",")
  return lines.slice(1).map((line) => {
    const cells = line.split(",")
    const row: Record<string, string> = {}
    header.forEach((key, index) => {
      row[key] = cells[index] ?? ""
    })
    return row as unknown as T
  })
}

export function parseIncidentsCsv(text: string): IncidentCsvRow[] {
  return parseCsv<IncidentCsvRow>(text)
}

export function parseTracksCsv(text: string): TrackCsvRow[] {
  return parseCsv<TrackCsvRow>(text)
}
