import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const { pathname } = useLocation();
  return (
    <nav className="navbar">
      <Link to="/" className="nav-logo">
        Sono<span>Lens</span>
      </Link>
      <ul className="nav-links">
        <li><Link to="/" className={pathname === "/" ? "active" : ""}>Home</Link></li>
        <li><Link to="/analyse" className={pathname === "/analyse" ? "active" : ""}>Analyse</Link></li>
      </ul>
      <Link to="/analyse" className="btn-primary">Get Started</Link>
    </nav>
  );
}