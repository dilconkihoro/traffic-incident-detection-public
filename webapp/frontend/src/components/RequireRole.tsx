import { Outlet } from "react-router-dom"
import { useAuth } from "../lib/AuthContext"
import { ROLE_LABELS } from "../lib/constants"
import type { Role } from "../lib/types"
import { ErrorBanner } from "./States"

export function RequireRole({ roles }: { roles: Role[] }) {
  const { role } = useAuth()
  if (role !== null && roles.includes(role)) return <Outlet />
  return (
    <div className="mx-auto max-w-2xl" data-testid="forbidden">
      <ErrorBanner
        message={`This page is not available to your role (${role ? ROLE_LABELS[role] : "unknown"}). Ask an administrator if you need access.`}
      />
    </div>
  )
}
