import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar({ user, onLogout }) {
  const { pathname } = useLocation();

  return (
    <nav className="navbar">
      <Link to="/home" className="nav-logo">
        Sono<span>Lens</span>
      </Link>

      <ul className="nav-links">
        <li><Link to="/home" className={pathname === "/home" ? "active" : ""}>Home</Link></li>
        <li><Link to="/analyse" className={pathname === "/analyse" ? "active" : ""}>Analyse</Link></li>
      </ul>

      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div className="nav-user">
          <div className="nav-avatar">{user?.name?.[0]?.toUpperCase() || "U"}</div>
          <span className="nav-user-name">{user?.name}</span>
        </div>
        <button
          className="btn-outline"
          style={{ padding: "8px 18px", fontSize: 13 }}
          onClick={onLogout}
        >
          Sign Out
        </button>
      </div>
    </nav>
  );
}