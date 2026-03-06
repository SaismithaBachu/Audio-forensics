// import React from "react";

// export default function PredictionCard({ scene, device }) {

//   return (

//     <div className="card">

//       <h2>Prediction Result</h2>

//       <div className="prediction">

//         <div>
//           <h4>Scene</h4>
//           <div className="scene">
//             {scene}
//           </div>
//         </div>

//         <div>
//           <h4>Device</h4>
//           <div className="device">
//             {device}
//           </div>
//         </div>

//       </div>

//     </div>

//   );

// }

import React from "react";

function ConfidenceGauge({ score }) {
  const pct = Math.min(Math.max(score, 0), 100);
  const angle = -180 + (pct / 100) * 180;
  const rad = (angle * Math.PI) / 180;
  const cx = 90, cy = 90, r = 68;
  const nx = cx + r * Math.cos(rad);
  const ny = cy + r * Math.sin(rad);
  const color = pct >= 70 ? "#3d2fa9" : pct >= 40 ? "#f5a623" : "#e8522a";
  const label = pct >= 70 ? "High Confidence" : pct >= 40 ? "Moderate" : "Low Confidence";

  return (
    <div className="gauge-wrap">
      <svg width="180" height="100" viewBox="0 0 180 100">
        <path d="M18 90 A72 72 0 0 1 162 90" fill="none" stroke="#ede9ff" strokeWidth="10" strokeLinecap="round" />
        <path d={`M18 90 A72 72 0 0 1 ${nx.toFixed(1)} ${ny.toFixed(1)}`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" />
        <circle cx={nx.toFixed(1)} cy={ny.toFixed(1)} r="6" fill={color} />
      </svg>
      <div className="gauge-score">{pct}%</div>
      <div className="gauge-sublabel" style={{ color }}>{label}</div>
    </div>
  );
}

export default function PredictionCard({ scene, device, sceneProbs, deviceProbs, filename, sampleRate, duration }) {
  const topSceneConf = sceneProbs ? Math.round(Math.max(...sceneProbs) * 100) : null;
  const topDeviceConf = deviceProbs ? Math.round(Math.max(...deviceProbs) * 100) : null;

  return (
    <>
      {/* Overview strip */}
      <div className="overview-strip">
        <div>
          <div className="os-label">Detected Scene</div>
          <div className="os-scene">{scene}</div>
          <div className="os-file">{filename || "Uploaded audio"}</div>
        </div>

        <div className="os-mid">
          <div className="os-badge">Scene Confidence</div>
          <div className="os-conf">{topSceneConf !== null ? `${topSceneConf}%` : "—"}</div>
          <div className="os-conf-label">
            {topSceneConf >= 70 ? "High confidence match" : topSceneConf >= 40 ? "Moderate match" : "Uncertain"}
          </div>
        </div>

        <div className="os-right">
          <div className="os-label">Recording Device</div>
          <div className="os-device-val">{device}</div>
          {topDeviceConf !== null && (
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, opacity: 0.7, marginTop: 4 }}>
              Device confidence: {topDeviceConf}%
            </div>
          )}
        </div>
      </div>

      {/* Stats row - all audio-relevant */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="sc-label">Acoustic Scene</div>
          <div className="sc-value indigo" style={{ textTransform: "capitalize", fontSize: 16 }}>{scene}</div>
        </div>
        <div className="stat-card">
          <div className="sc-label">Recording Device</div>
          <div className="sc-value" style={{ fontSize: 16 }}>{device}</div>
        </div>
        <div className="stat-card">
          <div className="sc-label">Scene Confidence</div>
          <div className="sc-value indigo">{topSceneConf !== null ? `${topSceneConf}%` : "—"}</div>
        </div>
        <div className="stat-card">
          <div className="sc-label">Device Confidence</div>
          <div className="sc-value">{topDeviceConf !== null ? `${topDeviceConf}%` : "—"}</div>
        </div>
      </div>

      {/* Confidence gauge */}
      <div className="card" style={{ gridColumn: "span 1" }}>
        <div className="card-title">Prediction Confidence</div>
        <ConfidenceGauge score={topSceneConf ?? 0} />
      </div>
    </>
  );
}