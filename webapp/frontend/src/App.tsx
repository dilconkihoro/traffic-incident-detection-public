import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { AuthProvider } from "./lib/AuthContext"
import { ProtectedLayout } from "./components/ProtectedLayout"
import Login from "./pages/Login"
import Upload from "./pages/Upload"
import Processing from "./pages/Processing"
import Dashboard from "./pages/Dashboard"
import RunDetail from "./pages/RunDetail"
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
            <Route path="/upload" element={<Upload />} />
            <Route path="/processing/:jobId" element={<Processing />} />
            <Route path="/runs/:id" element={<RunDetail />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
