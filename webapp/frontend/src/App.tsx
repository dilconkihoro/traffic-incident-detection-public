import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { AuthProvider } from "./lib/AuthContext"
import { ProtectedLayout } from "./components/ProtectedLayout"
import { RequireRole } from "./components/RequireRole"
import Login from "./pages/Login"
import Upload from "./pages/Upload"
import Processing from "./pages/Processing"
import Dashboard from "./pages/Dashboard"
import RunDetail from "./pages/RunDetail"
import Users from "./pages/Users"
import NotFound from "./pages/NotFound"

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<ProtectedLayout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/processing/:jobId" element={<Processing />} />
            <Route path="/runs/:id" element={<RunDetail />} />
            <Route element={<RequireRole roles={["admin", "operator"]} />}>
              <Route path="/upload" element={<Upload />} />
            </Route>
            <Route element={<RequireRole roles={["admin"]} />}>
              <Route path="/users" element={<Users />} />
            </Route>
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
