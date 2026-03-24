import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import "./index.css";

export default function App() {
  const [user, setUser] = useState(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    const session = localStorage.getItem("sono_session");
    if (session) setUser(JSON.parse(session));
    setChecked(true);
  }, []);

  const handleLogin = (userData) => setUser(userData);

  const handleLogout = () => {
    localStorage.removeItem("sono_session");
    setUser(null);
  };

  if (!checked) return null;

  return (
    <BrowserRouter>
      <Routes>
        {/* Root — redirect based on auth */}
        <Route path="/" element={
          user ? <Navigate to="/home" replace /> : <Navigate to="/login" replace />
        } />

        {/* Auth pages — no navbar */}
        <Route path="/login" element={
          user ? <Navigate to="/home" replace /> : <Login onLogin={handleLogin} />
        } />
        <Route path="/signup" element={
          user ? <Navigate to="/home" replace /> : <Signup onLogin={handleLogin} />
        } />

        {/* Protected pages — with navbar */}
        <Route path="/home" element={
          user
            ? <><Navbar user={user} onLogout={handleLogout} /><Home /></>
            : <Navigate to="/login" replace />
        } />
        <Route path="/analyse" element={
          user
            ? <><Navbar user={user} onLogout={handleLogout} /><Predict /></>
            : <Navigate to="/login" replace />
        } />
      </Routes>
    </BrowserRouter>
  );
}