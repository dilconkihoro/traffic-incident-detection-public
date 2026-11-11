import { useCallback, useEffect, useState, type FormEvent } from "react"
import { ApiError, createUser, listUsers, updateUser } from "../lib/api"
import { useAuth } from "../lib/AuthContext"
import { ROLE_DESCRIPTIONS, ROLE_LABELS } from "../lib/constants"
import { formatDateTime } from "../lib/format"
import type { Role, UserRow } from "../lib/types"
import { ErrorBanner, PageHeading, Spinner } from "../components/States"

const ROLES: Role[] = ["admin", "operator", "analyst"]
const MIN_PASSWORD_LENGTH = 8

function message(err: unknown, fallback: string): string {
  return err instanceof ApiError ? err.message : fallback
}

export default function Users() {
  const { username: me } = useAuth()
  const [users, setUsers] = useState<UserRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [form, setForm] = useState({ username: "", password: "", role: "operator" as Role })
  const [saving, setSaving] = useState(false)

  const reload = useCallback(() => {
    listUsers()
      .then(setUsers)
      .catch((err: unknown) => setError(message(err, "Failed to load users.")))
  }, [])

  useEffect(reload, [reload])

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError(null)
    setNotice(null)
    if (form.password.length < MIN_PASSWORD_LENGTH) {
      setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`)
      return
    }
    setSaving(true)
    try {
      const created = await createUser(form)
      setNotice(`Created ${created.username} as ${ROLE_LABELS[created.role]}.`)
      setForm({ username: "", password: "", role: form.role })
      reload()
    } catch (err) {
      setError(message(err, "Could not create the user."))
    } finally {
      setSaving(false)
    }
  }

  const change = async (user: UserRow, payload: Parameters<typeof updateUser>[1], done: string) => {
    setError(null)
    setNotice(null)
    try {
      await updateUser(user.id, payload)
      setNotice(done)
      reload()
    } catch (err) {
      setError(message(err, "Could not update the user."))
    }
  }

  const resetPassword = (user: UserRow) => {
    const password = window.prompt(`New password for ${user.username} (at least ${MIN_PASSWORD_LENGTH} characters):`)
    if (password === null) return
    void change(user, { password }, `Password reset for ${user.username}; their old sessions are signed out.`)
  }

  return (
    <div data-testid="users-page">
      <PageHeading
        title="Users"
        description="Create accounts and assign each person the role that matches what they need to do."
      />

      <div className="mb-6 grid gap-3 sm:grid-cols-3">
        {ROLES.map((role) => (
          <div key={role} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
            <p className="text-sm font-semibold text-slate-900">{ROLE_LABELS[role]}</p>
            <p className="mt-1 text-xs text-slate-500">{ROLE_DESCRIPTIONS[role]}</p>
          </div>
        ))}
      </div>

      <form
        onSubmit={submit}
        className="mb-6 grid gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:grid-cols-[1fr_1fr_auto_auto] sm:items-end"
        aria-label="Create user"
      >
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">Username</span>
          <input
            name="username"
            required
            maxLength={64}
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            className="w-full rounded-md border border-slate-300 px-3 py-2"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">Temporary password</span>
          <input
            name="password"
            type="password"
            required
            minLength={MIN_PASSWORD_LENGTH}
            autoComplete="new-password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="w-full rounded-md border border-slate-300 px-3 py-2"
          />
        </label>
        <label className="text-sm">
          <span className="mb-1 block font-medium text-slate-700">Role</span>
          <select
            name="role"
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value as Role })}
            className="w-full rounded-md border border-slate-300 bg-white px-3 py-2"
          >
            {ROLES.map((role) => (
              <option key={role} value={role}>
                {ROLE_LABELS[role]}
              </option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          disabled={saving}
          className="flex items-center justify-center gap-2 rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-60"
        >
          {saving && <Spinner className="h-4 w-4" />}
          Create user
        </button>
      </form>

      {error && (
        <div className="mb-4" data-testid="users-error">
          <ErrorBanner message={error} />
        </div>
      )}
      {notice && (
        <p className="mb-4 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800" data-testid="users-notice">
          {notice}
        </p>
      )}

      {users === null ? (
        <div className="flex justify-center py-12">
          <Spinner className="h-6 w-6 text-indigo-600" />
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-left text-sm" data-testid="users-table">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Username</th>
                <th scope="col" className="px-4 py-3 font-medium">Role</th>
                <th scope="col" className="px-4 py-3 font-medium">Status</th>
                <th scope="col" className="px-4 py-3 font-medium">Created</th>
                <th scope="col" className="px-4 py-3 font-medium">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((user) => {
                const locked = user.is_bootstrap || user.username === me
                return (
                  <tr key={user.id} data-testid={`user-row-${user.username}`}>
                    <td className="px-4 py-3 font-medium text-slate-900">
                      {user.username}
                      {user.is_bootstrap && <span className="ml-2 text-xs font-normal text-slate-400">(set in .env)</span>}
                    </td>
                    <td className="px-4 py-3">
                      <select
                        aria-label={`Role for ${user.username}`}
                        value={user.role}
                        disabled={locked}
                        onChange={(e) =>
                          void change(
                            user,
                            { role: e.target.value as Role },
                            `${user.username} is now ${ROLE_LABELS[e.target.value as Role]}.`,
                          )
                        }
                        className="rounded-md border border-slate-300 bg-white px-2 py-1 disabled:bg-slate-50 disabled:text-slate-400"
                      >
                        {ROLES.map((role) => (
                          <option key={role} value={role}>
                            {ROLE_LABELS[role]}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                          user.active ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
                        }`}
                      >
                        {user.active ? "Active" : "Disabled"}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-500">
                      {formatDateTime(user.created_at)}
                      {user.created_by && ` by ${user.created_by}`}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {!locked && (
                        <div className="flex justify-end gap-3">
                          <button
                            type="button"
                            onClick={() => resetPassword(user)}
                            className="font-medium text-indigo-600 hover:text-indigo-500"
                          >
                            Reset password
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              void change(
                                user,
                                { active: !user.active },
                                `${user.username} ${user.active ? "disabled" : "enabled"}.`,
                              )
                            }
                            className={user.active ? "font-medium text-rose-600 hover:text-rose-500" : "font-medium text-emerald-700 hover:text-emerald-600"}
                          >
                            {user.active ? "Disable" : "Enable"}
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
