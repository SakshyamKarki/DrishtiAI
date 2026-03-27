import { Link, useLocation, useNavigate } from "react-router-dom";
import "../styles/navbar.css"
import { useDispatch, useSelector } from "react-redux";
import { logoutThunk, selectUser } from "../features/auth/authSlice";
import { useEffect, useReducer, useRef, useState } from "react";

const Navbar = () => {
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const location = useLocation();
    const user = useSelector(selectUser);
    const [profileOpen, setProfileOpen] = useState(false);
    const dropdownRef = useRef(null);

    const navLinks = [
        {label:"Dashboard", path:"/dashboard"},
        {label:"Upload", path:"/upload"},
        {label:"History", path:"/history"},
    ];

    useEffect(()=>{
        const handleClickOutside = (e) =>{
            if(dropdownRef.current && !dropdownRef.current.contains(e.target)){
                setProfileOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return ()=> document.removeEventListener("mousedown", handleClickOutside);
    },[]);

    const handleLogout = async() =>{
        await dispatch(logoutThunk());
        navigate("/login");
    }

    const getInitials = (name) =>{
        if(!name) return "?";
        return name 
            .split(" ")
            .map((n)=>n[0])
            .join("")
            .toUpperCase()
            .slice(0, 2);
    };

    return (
        <nav className="navbar-shell">
             <Link to="/dashboard" className="navbar-pill navbar-left">
        <div className="navbar-logo-ring">
          <div className="navbar-logo-dot" />
        </div>
        <span className="navbar-logo-name">
          Drishti<span>AI</span>
        </span>
      </Link>

      {/* ── Center — Profile ── */}
      <div
        className="navbar-pill navbar-center"
        ref={dropdownRef}
        onClick={() => setProfileOpen((prev) => !prev)}
      >
        <div className="navbar-avatar">
          {getInitials(user?.name)}
        </div>
        <span className="navbar-username">{user?.name ?? "User"}</span>
        <span className={`navbar-chevron ${profileOpen ? "open" : ""}`}>▼</span>

        {/* dropdown */}
        {profileOpen && (
          <div className="navbar-dropdown">

            <div className="navbar-dropdown-header">
              <div className="navbar-dropdown-name">{user?.name}</div>
              <div className="navbar-dropdown-email">{user?.email}</div>
            </div>

            <div
              className="navbar-dropdown-item"
              onClick={(e) => {
                e.stopPropagation();
                setProfileOpen(false);
                navigate("/profile");
              }}
            >
              ◎ &nbsp; View profile
            </div>

            <div className="navbar-dropdown-divider" />

            <div
              className="navbar-dropdown-item danger"
              onClick={(e) => {
                e.stopPropagation();
                handleLogout();
              }}
            >
              ⏻ &nbsp; Log out
            </div>

          </div>
        )}
      </div>

      {/* ── Right — Nav Links ── */}
      <div className="navbar-pill navbar-right">
        {navLinks.map((link) => (
          <Link
            key={link.path}
            to={link.path}
            className={`navbar-link ${location.pathname === link.path ? "active" : ""}`}
          >
            {link.label}
          </Link>
        ))}
      </div>

        </nav>
    )
}

export default Navbar;