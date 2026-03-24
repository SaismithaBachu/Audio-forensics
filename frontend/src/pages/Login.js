import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (!email || !password) {
      setError("Please fill in all fields.");
      return;
    }

    setLoading(true);

    // Simulate auth — replace with real API call
    setTimeout(() => {
      const users = JSON.parse(localStorage.getItem("sono_users") || "[]");
      const user = users.find((u) => u.email === email);

      if (!user) {
        setError("No account found with this email. Please sign up.");
        setLoading(false);
        return;
      }

      if (user.password !== password) {
        setError("Incorrect password. Please try again.");
        setLoading(false);
        return;
      }

      localStorage.setItem("sono_session", JSON.stringify({ email: user.email, name: user.name }));
      onLogin({ email: user.email, name: user.name });
      navigate("/home");
    }, 800);
  };

  return (
    <div className="auth-page">
      {/* Left panel */}
      <div className="auth-left">
        <Link to="/" className="auth-logo">Sono<span>Lens</span></Link>

        <div className="auth-left-content">
          <div className="auth-wave-wrap">
            {Array.from({ length: 36 }, (_, i) => {
              const h = [30,55,70,40,80,50,25,65,45,85,55,30,72,60,38,58,78,35,62,48,28,68,44,82,54,22,74,58,36,54,76,34,60,46,30,50][i];
              return (
                <div key={i} className="auth-wave-bar" style={{
                  height: `${h}%`,
                  animationDelay: `${i * 0.05}s`,
                }} />
              );
            })}
          </div>
          <h2 className="auth-left-title">Decode any<br />acoustic environment</h2>
          <p className="auth-left-sub">
            Upload audio. Get forensic-grade scene and device predictions in seconds.
          </p>

          <div className="auth-chips">
            {["Airport","Metro","Park","Shopping Mall","Bus","Street"].map((s) => (
              <span key={s} className="auth-chip">{s}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="auth-right">
        <div className="auth-form-wrap">
          <div className="auth-form-header">
            <h1>Welcome back</h1>
            <p>Sign in to your SonoLens account</p>
          </div>

          {error && (
            <div className="auth-error">{error}</div>
          )}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="auth-field">
              <label>Email address</label>
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>

            <div className="auth-field">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <label>Password</label>
                <span className="auth-forgot">Forgot password?</span>
              </div>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </div>

            <button type="submit" className="auth-submit" disabled={loading}>
              {loading ? <span className="auth-spinner" /> : "Sign In"}
            </button>
          </form>

          <div className="auth-divider"><span>or</span></div>

          <p className="auth-switch">
            Don't have an account?{" "}
            <Link to="/signup">Create one free</Link>
          </p>
        </div>
      </div>
    </div>
  );
}