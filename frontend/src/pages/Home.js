// import React from "react";

// export default function Home(){

// return(

// <div>

// <div className="hero">

// <h1>Audio Scene Intelligence</h1>

// <p>
// Detect environments and recording devices using deep learning.
// </p>

// <button>
// Start Analysis
// </button>

// </div>

// </div>

// );

// }

import React from "react";
import { Link } from "react-router-dom";

const WAVE_HEIGHTS = [20,40,60,35,75,50,30,65,45,80,55,30,70,85,50,40,65,45,75,55,35,60,80,45,30,70,55,40,65,50,35,75,60,45,80,55,30,65,50,40];

function HeroVisual() {
  return (
    <div className="hero-visual">
      <div className="hv-top">
        <div>
          <div className="hv-label">Live Detection</div>
          <div className="hv-scene">Metro Station</div>
        </div>
        <div className="hv-badge">94% CONF</div>
      </div>

      <div className="hero-wave">
        {WAVE_HEIGHTS.map((h, i) => (
          <div key={i} className="hw-bar" style={{
            height: `${h}%`,
            animationDelay: `${i * 0.035}s`,
          }} />
        ))}
      </div>

      <div className="hv-pills">
        {["airport","bus","metro","park","mall","street"].map((s, i) => (
          <span key={s} className={`hv-pill${i === 2 ? " active" : ""}`}>{s}</span>
        ))}
      </div>

      <div className="hv-stats">
        <div>
          <div className="hv-stat-label">Device</div>
          <div className="hv-stat-val">Device A</div>
        </div>
        <div>
          <div className="hv-stat-label">Duration</div>
          <div className="hv-stat-val">10.0 s</div>
        </div>
        <div>
          <div className="hv-stat-label">Signal</div>
          <div className="hv-stat-val indigo">MATCH</div>
        </div>
      </div>
    </div>
  );
}

const FEATURES = [
  { icon: "🎙️", title: "Mel-Spectrogram Analysis", tag: "Signal Processing", desc: "Every upload is converted to a 128-bin mel-spectrogram — the same frequency representation your ear uses." },
  { icon: "🧠", title: "Multi-Task CRNN", tag: "Deep Learning", desc: "One model simultaneously predicts acoustic scene and recording device from a single forward pass." },
  { icon: "📡", title: "10 Acoustic Scenes", tag: "Classification", desc: "Airport, bus, metro, park, public square, shopping mall, and more — classified in under a second." },
  { icon: "🎚️", title: "Waveform Visualiser", tag: "Audio Preview", desc: "Interactive waveform lets you play, scrub, and inspect your audio before and after analysis." },
  { icon: "📊", title: "Confidence Distribution", tag: "Explainability", desc: "Full probability breakdown across all scene classes — not just the top result." },
  { icon: "🔬", title: "Device Fingerprinting", tag: "Forensics", desc: "Identifies which recording device captured the audio — useful for forensic workflows." },
];

const STEPS = [
  { n: "01", t: "Upload Audio", d: "Drag & drop any WAV, MP3, or FLAC file. The backend accepts any standard audio format." },
  { n: "02", t: "Feature Extraction", d: "Backend computes a mel-spectrogram at 44.1 kHz and pads or trims to a fixed 320-frame window." },
  { n: "03", t: "Run CRNN Model", d: "The multi-task model processes the spectrogram and outputs scene and device probability vectors." },
  { n: "04", t: "Explore Results", d: "Interactive dashboard shows scene, device, confidence gauge, waveform, and full probability charts." },
];

export default function Home() {
  return (
    <div>
      {/* HERO */}
      <div style={{ background: "var(--bg-white)", borderBottom: "1px solid var(--border)" }}>
        <div className="hero">
          <div>
            <div className="hero-badge">
              <span className="hero-badge-dot" />
              AI-Powered Audio Scene Intelligence
            </div>
            <h1 className="hero-title">
              Decode Audio<br />with <span className="accent">Precision</span>
            </h1>
            <p className="hero-sub">
              SonoLens uses deep learning to identify acoustic environments and
              recording devices from raw audio. Upload a file — get forensic-grade
              scene and device analysis in seconds.
            </p>
            <div className="hero-actions">
              <Link to="/analyse" className="btn-primary">Start Analysing</Link>
              <a href="#features" className="btn-outline">Learn More</a>
            </div>
          </div>
          <HeroVisual />
        </div>
      </div>

      {/* FEATURES */}
      <div id="features" className="section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 52 }}>
          <div>
            <div className="section-tag">What We Offer</div>
            <h2 className="section-title" style={{ marginBottom: 8 }}>
              Everything you need<br />to analyse smarter
            </h2>
            <p style={{ fontSize: 15, color: "var(--text-3)" }}>6 powerful tools built into one platform</p>
          </div>
          <Link to="/analyse" className="btn-primary" style={{ whiteSpace: "nowrap" }}>
            Try It Now →
          </Link>
        </div>

        <div className="features-grid">
          {FEATURES.map((f, i) => (
            <div className="feature-card" key={i}>
              {/* Top: icon + tag */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
                <div className="feature-icon-wrap">{f.icon}</div>
                <span className="feat-tag">{f.tag}</span>
              </div>

              {/* Title */}
              <h3>{f.title}</h3>

              {/* Divider */}
              <div style={{ borderTop: "1px solid var(--border)", margin: "16px 0" }} />

              {/* Description + arrow */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 12 }}>
                <p>{f.desc}</p>
                <div className="feat-arrow">→</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="how-section">
        <div className="how-inner">
          <div className="section-tag">How It Works</div>
          <h2 className="section-title" style={{ marginBottom: 0 }}>
            Four steps to acoustic insight
          </h2>
          <p style={{ fontSize: 16, color: "var(--text-3)", marginTop: 8, marginBottom: 56 }}>
            Just 4 easy and quick steps
          </p>

          <div className="how-split">
            {/* Left — visual */}
            <div className="how-visual">
              <div className="how-visual-inner">
                {/* Animated waveform illustration */}
                <div className="how-wave-wrap">
                  <div className="how-wave">
                    {Array.from({ length: 48 }, (_, i) => {
                      const heights = [20,40,65,35,80,50,30,70,45,85,55,25,75,60,40,55,80,35,65,50,30,70,45,85,55,20,75,60,38,55,78,35,62,48,28,68,44,82,54,22,74,58,36,54,76,34,60,46];
                      return (
                        <div key={i} className="how-wave-bar" style={{
                          height: `${heights[i]}%`,
                          animationDelay: `${i * 0.05}s`,
                        }} />
                      );
                    })}
                  </div>
                  <div className="how-wave-label">Audio Signal</div>
                </div>

                {/* Arrow pointing right */}
                <div className="how-arrows">
                  <span>›</span><span>›</span><span>›</span>
                </div>

                {/* Mel spectrogram mini */}
                <div className="how-mel-mini">
                  <div className="how-mel-grid">
                    {Array.from({ length: 120 }, (_, i) => {
                      const row = Math.floor(i / 15);
                      const intensity = Math.max(0.1, 0.9 - row * 0.07 + (Math.random() * 0.3));
                      return (
                        <div key={i} className="how-mel-cell" style={{
                          opacity: intensity,
                          background: intensity > 0.6 ? "var(--indigo)" : intensity > 0.35 ? "#7b6fd4" : "#c4b9f5"
                        }} />
                      );
                    })}
                  </div>
                  <div className="how-mel-label">Mel Spectrogram</div>
                </div>
              </div>

              {/* Decorative pattern block */}
              <div className="how-deco" />
            </div>

            {/* Right — steps */}
            <div className="how-steps-list">
              {STEPS.map((s, i) => (
                <div className="how-step-item" key={s.n}>
                  <div className="how-step-num">{i + 1}</div>
                  <div className="how-step-content">
                    <h4>{s.t}</h4>
                    <p>{s.d}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="cta-wrap">
        <div className="cta-inner">
          <h2>Ready to analyse your first audio?</h2>
          <p>Upload any audio file and get a full scene and device prediction in seconds — no setup required.</p>
          <div className="cta-actions">
            <Link to="/analyse" className="btn-primary">Open Analysis Dashboard →</Link>
          </div>
        </div>
      </div>

      {/* FOOTER */}
      <footer className="footer">
        <div className="footer-logo">Sono<span>Lens</span></div>
        <div className="footer-meta">Built by SaismithaBachu · Open Source · GitHub</div>
        <div className="footer-meta">For research purposes only. Not legal forensic advice.</div>
      </footer>
    </div>
  );
}