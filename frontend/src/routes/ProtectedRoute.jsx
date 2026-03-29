import { useSelector } from "react-redux";
import { Navigate } from "react-router-dom";
import {
  selectAuthLoading,
  selectIsAuthenticated,
} from "../features/auth/authSlice";

const ProtectedRoute = ({ children }) => {
  const isAuthenticated = useSelector(selectIsAuthenticated);
  const isLoading = useSelector(selectAuthLoading);

  if (isLoading) return null;

  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

export default ProtectedRoute;
