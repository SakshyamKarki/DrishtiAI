import { Route, Routes } from "react-router-dom"
import ProtectedRoute from "./ProtectedRoute"
import Home from "../pages/Home"

const AppRoutes = () =>{
    return (
        <Routes>
            {/* Public routes */}
            <Route path="/login" element={}/>
            <Route path="/register" element={}/>

            {/* Protected routes */}
            <Route path="/" element={
                <ProtectedRoute>
                    <Home/>
                </ProtectedRoute>
            }/>
        </Routes>
    )
}