import { useSelector } from "react-redux";
import { Navigate } from "react-router-dom";
import { selectIsAuthenticated } from "../features/auth/authSlice";

const ProtectedRoute = ({children}) =>{
    const isAuthenticated = useSelector(selectIsAuthenticated);

    if(!isAuthenticated){
        return <Navigate to="/login" replace/>
    }

    return children;
};

export default ProtectedRoute;

