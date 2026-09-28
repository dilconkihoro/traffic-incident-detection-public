import { NavLink, useNavigate } from "react-router-dom"
import { useAuth } from "../lib/AuthContext"

const navLinkClasses = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-3 py-2 text-sm font-medium transition-colors ${
    isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
  }`

export function Header() {
  const { logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate("/login", { replace: true })
  }

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <NavLink to="/dashboard" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white">
            TW
          </span>
          <span className="text-base font-semibold text-slate-900">TrafficWatch</span>
        </NavLink>

        <nav className="flex items-center gap-1.5">
          <NavLink to="/dashboard" className={navLinkClasses} end>
            Dashboard
          </NavLink>
          <NavLink to="/upload" className={navLinkClasses}>
            New run
          </NavLink>
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
