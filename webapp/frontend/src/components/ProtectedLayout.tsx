import { Navigate, Outlet, useLocation } from "react-router-dom"
import { useAuth } from "../lib/AuthContext"
import { FullPageSpinner } from "./States"
import { Header } from "./Header"

export function ProtectedLayout() {
  const { status } = useAuth()
  const location = useLocation()

  if (status === "checking") {
    return <FullPageSpinner />
  }

  if (status === "unauthenticated") {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
    </div>
  )
}
