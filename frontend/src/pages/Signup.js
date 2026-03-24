import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Signup({ onLogin }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (!name || !email || !password || !confirm) {
      setError("Please fill in all fields.");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);

    setTimeout(() => {
      const users = JSON.parse(localStorage.getItem("sono_users") || "[]");

      if (users.find((u) => u.email === email)) {
        setError("An account with this email already exists. Please log in.");
        setLoading(false);
        return;
      }

      const newUser = { name, email, password };
      users.push(newUser);
      localStorage.setItem("sono_users", JSON.stringify(users));
      localStorage.setItem("sono_session", JSON.stringify({ email, name }));

      onLogin({ email, name });
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
          <h2 className="auth-left-title">Start analysing<br />audio instantly</h2>
          <p className="auth-left-sub">
            Create your free account and get forensic-grade acoustic scene and device detection.
          </p>

          <div className="auth-stats">
            <div className="auth-stat">
              <div className="auth-stat-val">8</div>
              <div className="auth-stat-label">Scene Classes</div>
            </div>
            <div className="auth-stat-divider" />
            <div className="auth-stat">
              <div className="auth-stat-val">&lt;1s</div>
              <div className="auth-stat-label">Prediction Time</div>
            </div>
            <div className="auth-stat-divider" />
            <div className="auth-stat">
              <div className="auth-stat-val">CRNN</div>
              <div className="auth-stat-label">Model Architecture</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right panel */}
      <div className="auth-right">
        <div className="auth-form-wrap">
          <div className="auth-form-header">
            <h1>Create account</h1>
            <p>Join SonoLens — it's free</p>
          </div>

          {error && (
            <div className="auth-error">{error}</div>
          )}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="auth-field">
              <label>Full name</label>
              <input
                type="text"
                placeholder="Your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoComplete="name"
              />
            </div>

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
              <label>Password</label>
              <input
                type="password"
                placeholder="Min. 6 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
              />
            </div>

            <div className="auth-field">
              <label>Confirm password</label>
              <input
                type="password"
                placeholder="Repeat password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
              />
            </div>

            {/* Password strength indicator */}
            {password && (
              <div className="auth-strength">
                {["weak","fair","good","strong"].map((lvl, i) => (
                  <div key={i} className={`auth-strength-bar ${
                    password.length >= (i + 1) * 3 ? "active" : ""
                  }`} />
                ))}
                <span className="auth-strength-label">
                  {password.length < 4 ? "Weak" : password.length < 7 ? "Fair" : password.length < 10 ? "Good" : "Strong"}
                </span>
              </div>
            )}

            <button type="submit" className="auth-submit" disabled={loading}>
              {loading ? <span className="auth-spinner" /> : "Create Account"}
            </button>
          </form>

          <div className="auth-divider"><span>or</span></div>

          <p className="auth-switch">
            Already have an account?{" "}
            <Link to="/login">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}