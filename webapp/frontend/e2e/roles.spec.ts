import { randomBytes } from "node:crypto"
import { existsSync, mkdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { expect, test, type Browser, type Page } from "@playwright/test"

// One walk through the dashboard as each role, in order: the administrator creates the other
// accounts, the operator processes a real UA-DETRAC clip, the analyst reviews it read-only, and
// the administrator then disables an account. Screenshots are the figure F5.14 series (see the
// evidence list in the documentation plan).
const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, "../../..")
const evidenceDir = path.join(repoRoot, "docs", "evidence")
const sampleClip = path.join(repoRoot, "data", "samples", "MVI_20011_short.mp4")

const admin = { username: process.env.E2E_ADMIN_USERNAME!, password: process.env.E2E_ADMIN_PASSWORD! }
const operator = { username: "e2e-operator", password: randomBytes(9).toString("hex") }
const analyst = { username: "e2e-analyst", password: randomBytes(9).toString("hex") }

let finishedRunUrl = ""

async function evidence(page: Page, name: string) {
  mkdirSync(evidenceDir, { recursive: true })
  await page.screenshot({ path: path.join(evidenceDir, `${name}.png`), fullPage: true })
}

async function signIn(browser: Browser, user: { username: string; password: string }): Promise<Page> {
  const page = await (await browser.newContext()).newPage()
  await page.goto("/login")
  await page.getByLabel(/username/i).fill(user.username)
  await page.getByLabel(/password/i).fill(user.password)
  await page.getByRole("button", { name: /sign in|log in/i }).click()
  await expect(page).toHaveURL(/\/dashboard$/)
  return page
}

test.describe.serial("role-based dashboard", () => {
  test("a wrong password is refused without saying which field was wrong", async ({ page }) => {
    await page.goto("/login")
    await page.getByLabel(/username/i).fill(admin.username)
    await page.getByLabel(/password/i).fill("not-the-password")
    await page.getByRole("button", { name: /sign in|log in/i }).click()
    await expect(page.getByText("Invalid username or password.")).toBeVisible()
    await evidence(page, "F5.14a_login_refused")
  })

  test("the administrator sees the admin dashboard and creates an operator and an analyst", async ({ browser }) => {
    const page = await signIn(browser, admin)
    await expect(page.getByTestId("dashboard-admin")).toBeVisible()
    await expect(page.getByRole("heading", { name: "Administrator dashboard" })).toBeVisible()
    await expect(page.getByRole("navigation").getByRole("link", { name: "Users" })).toBeVisible()
    await expect(page.getByTestId("current-user")).toContainText("Administrator")
    await evidence(page, "F5.14b_dashboard_admin")

    await page.getByRole("link", { name: "Manage users" }).click()
    await expect(page.getByTestId("users-page")).toBeVisible()
    for (const [user, role] of [
      [operator, "Traffic operator"],
      [analyst, "Analyst"],
    ] as const) {
      const form = page.getByRole("form", { name: "Create user" })
      await form.getByLabel("Username").fill(user.username)
      await form.getByLabel("Temporary password").fill(user.password)
      await form.getByLabel("Role").selectOption({ label: role })
      await form.getByRole("button", { name: "Create user" }).click()
      await expect(page.getByTestId("users-notice")).toContainText(`Created ${user.username} as ${role}`)
      await expect(page.getByTestId(`user-row-${user.username}`)).toBeVisible()
    }

    const form = page.getByRole("form", { name: "Create user" })
    await form.getByLabel("Username").fill(operator.username)
    await form.getByLabel("Temporary password").fill("another-password-1")
    await form.getByRole("button", { name: "Create user" }).click()
    await expect(page.getByTestId("users-error")).toContainText("already exists")

    await expect(page.getByTestId(`user-row-${admin.username}`).getByRole("combobox")).toBeDisabled()
    await evidence(page, "F5.14c_users_admin")
    await page.context().close()
  })

  test("the operator processes a real clip and sees vehicles with anonymous identifiers", async ({ browser }) => {
    test.skip(!existsSync(sampleClip), `needs ${sampleClip} (docs/08 section 10c)`)
    const page = await signIn(browser, operator)
    await expect(page.getByTestId("dashboard-operator")).toBeVisible()
    await expect(page.getByRole("navigation").getByRole("link", { name: "Users" })).toHaveCount(0)
    await evidence(page, "F5.14d_dashboard_operator")

    await page.goto("/users")
    await expect(page.getByTestId("forbidden")).toBeVisible()

    await page.goto("/upload")
    await expect(page.getByText("Detection model")).toBeVisible()
    await expect(page.getByText(/Fine-tuned for this project|Not available/)).toBeVisible()
    await page.locator('input[type="file"]').setInputFiles(sampleClip)
    await expect(page.getByText("MVI_20011_short.mp4")).toBeVisible()
    await evidence(page, "F5.14e_upload_operator")
    await page.getByRole("button", { name: "Start processing" }).click()

    await expect(page).toHaveURL(/\/processing\//, { timeout: 60 * 1000 })
    await expect(page.getByRole("button", { name: "Cancel run" })).toBeVisible()
    // Capture the live view part-way through; a fast machine may finish first, which is fine.
    const framesDone = async () => {
      if (!page.url().includes("/processing/")) return Number.POSITIVE_INFINITY
      const text = (await page.getByText(/\d+ \/ 75 frames/).textContent().catch(() => null)) ?? "0"
      return Number(text.split(" ")[0])
    }
    await expect.poll(framesDone, { timeout: 5 * 60 * 1000, intervals: [1000] }).toBeGreaterThanOrEqual(30)
    if (page.url().includes("/processing/")) await evidence(page, "F5.14f_processing_operator")

    await expect(page).toHaveURL(/\/runs\//, { timeout: 8 * 60 * 1000 })
    finishedRunUrl = page.url()
    const vehicles = page.getByTestId("vehicles-table")
    await expect(vehicles).toBeVisible()
    await expect(vehicles.locator("tbody tr").first()).toContainText(/VH-[A-Z2-9]{4,}/)
    await expect(page.getByText(/Vehicle(s)? VH-|No incidents flagged/).first()).toBeVisible()
    await evidence(page, "F5.14g_results_vehicles")
    await page.context().close()
  })

  test("the analyst reviews runs read-only and cannot start or manage anything", async ({ browser }) => {
    const page = await signIn(browser, analyst)
    await expect(page.getByTestId("dashboard-analyst")).toBeVisible()
    await expect(page.getByRole("link", { name: "New run" })).toHaveCount(0)
    await expect(page.getByRole("navigation").getByRole("link", { name: "Users" })).toHaveCount(0)
    if (finishedRunUrl) {
      await expect(page.getByTestId("runs-table")).toContainText(operator.username)
    }
    await evidence(page, "F5.14h_dashboard_analyst")

    if (finishedRunUrl) {
      await page.goto(finishedRunUrl)
      await expect(page.getByTestId("vehicles-table")).toBeVisible()
      await expect(page.getByRole("link", { name: "Download CSV" }).first()).toBeVisible()
    }

    await page.goto("/upload")
    await expect(page.getByTestId("forbidden")).toBeVisible()
    await evidence(page, "F5.14i_upload_refused_analyst")

    const refused = await page.request.post("/api/users", {
      data: { username: "sneaky", password: "long-enough-1", role: "admin" },
    })
    expect(refused.status()).toBe(403)
    await page.context().close()
  })

  test("disabling an account signs that user out on their next request", async ({ browser }) => {
    const analystPage = await signIn(browser, analyst)
    const adminPage = await signIn(browser, admin)

    await adminPage.goto("/users")
    await adminPage.getByTestId(`user-row-${analyst.username}`).getByRole("button", { name: "Disable" }).click()
    await expect(adminPage.getByTestId("users-notice")).toContainText(`${analyst.username} disabled`)
    await expect(adminPage.getByTestId(`user-row-${analyst.username}`)).toContainText("Disabled")
    await evidence(adminPage, "F5.14j_user_disabled_admin")

    await analystPage.reload()
    await expect(analystPage).toHaveURL(/\/login$/)
  })
})
