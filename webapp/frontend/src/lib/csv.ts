import type { IncidentCsvRow, TrackCsvRow, VehicleCsvRow } from "./types"

// The producers (src/trafficwatch/incident_log.py) write with Python's csv module, which quotes a
// field only when it contains a comma, quote or newline; vehicles.csv descriptions do contain
// commas. This reads that format: quoted fields with "" as an escaped quote, unquoted otherwise.
function splitLine(line: string): string[] {
  const cells: string[] = []
  let cell = ""
  let quoted = false
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i]
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        cell += '"'
        i += 1
      } else if (char === '"') {
        quoted = false
      } else {
        cell += char
      }
    } else if (char === '"') {
      quoted = true
    } else if (char === ",") {
      cells.push(cell)
      cell = ""
    } else {
      cell += char
    }
  }
  cells.push(cell)
  return cells
}

function parseCsv<T>(text: string): T[] {
  const lines = text.split("\n").filter((line) => line.length > 0)
  if (lines.length === 0) return []

  const header = splitLine(lines[0])
  return lines.slice(1).map((line) => {
    const cells = splitLine(line)
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

export function parseVehiclesCsv(text: string): VehicleCsvRow[] {
  return parseCsv<VehicleCsvRow>(text)
}
