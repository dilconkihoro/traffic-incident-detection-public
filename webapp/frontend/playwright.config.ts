import { randomBytes } from "node:crypto"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { defineConfig, devices } from "@playwright/test"

// End-to-end tests for the role-based dashboard. They start their own backend (temporary data,
// credentials generated per run) and their own dev server on separate ports, so they can run
// while a normal dashboard session is open. See webapp/README.md, "End-to-end tests".
const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, "../..")
const apiPort = Number(process.env.E2E_API_PORT ?? 8011)
const webPort = Number(process.env.E2E_WEB_PORT ?? 5181)

// Config files are evaluated by the runner and by each worker; generate once and hand the same
// values to both through the environment.
process.env.E2E_WORK_DIR ??= mkdtempSync(path.join(tmpdir(), "tw-e2e-"))
process.env.E2E_SECRET_KEY ??= randomBytes(32).toString("hex")
process.env.E2E_ADMIN_USERNAME ??= "e2e-admin"
process.env.E2E_ADMIN_PASSWORD ??= randomBytes(12).toString("hex")
process.env.E2E_API_PORT = String(apiPort)

const python =
  process.env.E2E_PYTHON ??
  (process.platform === "win32"
    ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, ".venv", "bin", "python"))

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 10 * 60 * 1000,
  expect: { timeout: 15 * 1000 },
  reporter: [["list"]],
  outputDir: path.join(process.env.E2E_WORK_DIR, "results"),
  use: {
    baseURL: `http://127.0.0.1:${webPort}`,
    viewport: { width: 1280, height: 800 },
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } } }],
  webServer: [
    {
      command: `"${python}" e2e/serve_backend.py`,
      url: `http://127.0.0.1:${apiPort}/api/me`,
      // /api/me answers 401 without a session, which proves the server is up.
      ignoreHTTPSErrors: true,
      reuseExistingServer: false,
      timeout: 120 * 1000,
      env: { ...process.env } as Record<string, string>,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${webPort} --strictPort`,
      url: `http://127.0.0.1:${webPort}/login`,
      reuseExistingServer: false,
      timeout: 120 * 1000,
      env: { ...process.env, TW_API_URL: `http://127.0.0.1:${apiPort}` } as Record<string, string>,
    },
  ],
})
