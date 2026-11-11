import { NavLink, useNavigate } from "react-router-dom"
import { useAuth } from "../lib/AuthContext"
import { ROLE_LABELS, canRunDetection } from "../lib/constants"

const navLinkClasses = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-3 py-2 text-sm font-medium transition-colors ${
    isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
  }`

export function Header() {
  const { logout, username, role } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate("/login", { replace: true })
  }

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <NavLink to="/dashboard" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white">
            TW
          </span>
          <span className="text-base font-semibold text-slate-900">TrafficWatch</span>
        </NavLink>

        <nav className="flex flex-wrap items-center gap-1.5" aria-label="Main">
          <NavLink to="/dashboard" className={navLinkClasses} end>
            Dashboard
          </NavLink>
          {canRunDetection(role) && (
            <NavLink to="/upload" className={navLinkClasses}>
              New run
            </NavLink>
          )}
          {role === "admin" && (
            <NavLink to="/users" className={navLinkClasses}>
              Users
            </NavLink>
          )}
          {username && role && (
            <span className="ml-2 hidden text-right text-xs leading-tight text-slate-500 sm:block" data-testid="current-user">
              <span className="block font-medium text-slate-900">{username}</span>
              {ROLE_LABELS[role]}
            </span>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="ml-2 rounded-md border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900"
          >
            Log out
          </button>
        </nav>
      </div>
    </header>
  )
}
