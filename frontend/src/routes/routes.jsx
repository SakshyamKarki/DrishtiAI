import { Navigate } from "react-router-dom";
import ProtectedRoute from "./ProtectedRoute";
import RegisterPage from "../pages/RegisterPage";
import LoginPage from "../pages/LoginPage";
import HistoryPage from "../pages/HistoryPage";
import App from "../App"
import LandingPage from "../pages/LandingPage";
import Dashboard from "../pages/Dashboard";
import UploadPage from "../pages/UploadPage";

const routes = [
  //public routes
  { path: "/login", element: <LoginPage /> },
  { path: "/register", element: <RegisterPage /> },
  { path: "/", element: <LandingPage /> },
  //protected routes
  {
    element: (
    //   <ProtectedRoute>
        <App />
    //   </ProtectedRoute>
    ),
    children: [
      {
        path: "/dashboard",
        element: <Dashboard />,
      },
      {
        path: "/upload",
        element: <UploadPage />,
      },
      {
        path: "/history",
        element: <HistoryPage />,
      },
    ],
  },

  //defaults
  { path: "*", element: <Navigate to="/" replace /> },
];

export default routes;
