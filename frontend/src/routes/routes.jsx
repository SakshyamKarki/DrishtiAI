import { Navigate } from "react-router-dom";
import ProtectedRoute from "./ProtectedRoute";
import RegisterPage from "../pages/RegisterPage";
import LoginPage from "../pages/LoginPage";
import Home from "../pages/Home";
import Detect from "../pages/Detect";
import About from "../pages/About";
import HistoryPage from "../pages/HistoryPage";
import App from "../App"
import LandingPage from "../pages/LandingPage";

const routes = [
  //public routes
  { path: "/login", element: <LoginPage /> },
  { path: "/register", element: <RegisterPage /> },
  { path: "/", element: <LandingPage /> },
  //protected routes
  {
    element: (
      <ProtectedRoute>
        <App />
      </ProtectedRoute>
    ),
    children: [
      {
        path: "/",
        element: <Home />,
      },
      {
        path: "/detection",
        element: <Detect />,
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
