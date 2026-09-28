import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"
import { listRuns, login as apiLogin, logout as apiLogout, setUnauthorizedHandler } from "./api"

type AuthStatus = "checking" | "authenticated" | "unauthenticated"

interface AuthContextValue {
  status: AuthStatus
  username: string | null
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("checking")
  const [username, setUsername] = useState<string | null>(null)

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setStatus("unauthenticated")
      setUsername(null)
    })
    return () => setUnauthorizedHandler(null)
  }, [])

  // There is no dedicated "whoami" endpoint; a cheap protected call doubles as the session
  // check that runs once when the app loads (a page refresh loses no more than this ping).
  useEffect(() => {
    let cancelled = false
    listRuns()
      .then(() => {
        if (!cancelled) setStatus("authenticated")
      })
      .catch(() => {
        if (!cancelled) setStatus("unauthenticated")
      })
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (user: string, password: string) => {
    const response = await apiLogin(user, password)
    setUsername(response.username)
    setStatus("authenticated")
  }, [])

  const logout = useCallback(async () => {
    try {
      await apiLogout()
    } finally {
      setUsername(null)
      setStatus("unauthenticated")
    }
  }, [])

  return (
    <AuthContext.Provider value={{ status, username, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
