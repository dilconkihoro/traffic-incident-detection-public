import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"
import { getMe, login as apiLogin, logout as apiLogout, setUnauthorizedHandler } from "./api"
import type { Role } from "./types"

type AuthStatus = "checking" | "authenticated" | "unauthenticated"

interface AuthContextValue {
  status: AuthStatus
  username: string | null
  role: Role | null
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("checking")
  const [username, setUsername] = useState<string | null>(null)
  const [role, setRole] = useState<Role | null>(null)

  const signedOut = useCallback(() => {
    setStatus("unauthenticated")
    setUsername(null)
    setRole(null)
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(signedOut)
    return () => setUnauthorizedHandler(null)
  }, [signedOut])

  // Restores the session after a page refresh; the server re-checks the account on every
  // request, so a role changed by an administrator shows up here on the next load.
  useEffect(() => {
    let cancelled = false
    getMe()
      .then((me) => {
        if (cancelled) return
        setUsername(me.username)
        setRole(me.role)
        setStatus("authenticated")
      })
      .catch(() => {
        if (!cancelled) signedOut()
      })
    return () => {
      cancelled = true
    }
  }, [signedOut])

  const login = useCallback(async (user: string, password: string) => {
    const response = await apiLogin(user, password)
    setUsername(response.username)
    setRole(response.role)
    setStatus("authenticated")
  }, [])

  const logout = useCallback(async () => {
    try {
      await apiLogout()
    } finally {
      signedOut()
    }
  }, [signedOut])

  return (
    <AuthContext.Provider value={{ status, username, role, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
