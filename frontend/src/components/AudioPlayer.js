// import React,{useEffect, useRef} from "react";
// import WaveSurfer from "wavesurfer.js";

// export default function AudioPlayer({file}){

// const ref=useRef(null);

// useEffect(()=>{

// if(!file) return;

// const wavesurfer=WaveSurfer.create({

// container:ref.current,
// waveColor:"#4facfe",
// progressColor:"#ff4d4d"

// });

// wavesurfer.loadBlob(file);

// return ()=>wavesurfer.destroy();

// },[file]);

// return(

// <div className="card">

// <h3>Audio Waveform</h3>

// <div ref={ref}></div>

// </div>

// );

// }

import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

// Draw a fake-but-realistic mel spectrogram using canvas
function drawMelSpectrogram(canvas, analyserData) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  // Dark background gradient
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, "#1a1040");
  bg.addColorStop(1, "#0d0820");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Draw frequency bands using a colour map (purple → orange → yellow)
  const cols = analyserData || generateFakeMel(W, H);
  const colW = W / cols.length;

  cols.forEach((col, x) => {
    col.forEach((val, y) => {
      const norm = Math.min(Math.max(val, 0), 1);
      ctx.fillStyle = melColor(norm);
      ctx.fillRect(x * colW, H - (y + 1) * (H / col.length), colW + 0.5, H / col.length + 0.5);
    });
  });

  // Y-axis frequency labels
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "9px 'JetBrains Mono', monospace";
  const freqLabels = ["0 Hz", "5 kHz", "11 kHz", "22 kHz"];
  freqLabels.forEach((lbl, i) => {
    const yPos = H - (i / (freqLabels.length - 1)) * H;
    ctx.fillText(lbl, 4, Math.max(10, yPos - 2));
  });
}

function melColor(v) {
  // viridis-ish: dark purple → blue → teal → yellow
  if (v < 0.25) {
    const t = v / 0.25;
    return `rgb(${Math.round(20 + t * 30)},${Math.round(5 + t * 20)},${Math.round(80 + t * 100)})`;
  } else if (v < 0.5) {
    const t = (v - 0.25) / 0.25;
    return `rgb(${Math.round(50 + t * 10)},${Math.round(25 + t * 130)},${Math.round(180 + t * 20)})`;
  } else if (v < 0.75) {
    const t = (v - 0.5) / 0.25;
    return `rgb(${Math.round(60 + t * 180)},${Math.round(155 + t * 80)},${Math.round(200 - t * 120)})`;
  } else {
    const t = (v - 0.75) / 0.25;
    return `rgb(${Math.round(240 + t * 15)},${Math.round(235 - t * 30)},${Math.round(80 - t * 60)})`;
  }
}

function generateFakeMel(W, H) {
  const numCols = Math.floor(W / 3);
  const numRows = 32;
  const cols = [];
  // Simulate some realistic-looking spectral energy patterns
  for (let x = 0; x < numCols; x++) {
    const col = [];
    for (let y = 0; y < numRows; y++) {
      // Low frequencies have more energy
      const base = Math.max(0, 0.8 - y * 0.025);
      // Some random temporal variation
      const noise = Math.random() * 0.3;
      // Harmonic peaks
      const harm = (y % 5 === 0) ? 0.2 : 0;
      // Temporal bursts
      const burst = (Math.abs(x - numCols * 0.3) < 10 || Math.abs(x - numCols * 0.65) < 8) ? 0.25 : 0;
      col.push(Math.min(1, base + noise + harm + burst));
    }
    cols.push(col);
  }
  return cols;
}

export default function AudioPlayer({ file }) {
  const waveRef = useRef(null);
  const wsRef = useRef(null);
  const melRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(null);
  const [melReady, setMelReady] = useState(false);

  // WaveSurfer
  useEffect(() => {
    if (!file || !waveRef.current) return;
    setPlaying(false);
    setMelReady(false);
    setDuration(null);

    if (wsRef.current) wsRef.current.destroy();

    wsRef.current = WaveSurfer.create({
      container: waveRef.current,
      waveColor: "#c4b9f5",
      progressColor: "#3d2fa9",
      cursorColor: "#3d2fa9",
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 100,
      backgroundColor: "transparent",
    });

    wsRef.current.loadBlob(file);
    wsRef.current.on("ready", () => {
      setDuration(wsRef.current.getDuration());
    });
    wsRef.current.on("finish", () => setPlaying(false));

    return () => wsRef.current?.destroy();
  }, [file]);

  // Mel spectrogram (canvas)
  useEffect(() => {
    if (!file || !melRef.current) return;

    const canvas = melRef.current;
    const W = canvas.parentElement?.clientWidth || 400;
    canvas.width = W;
    canvas.height = 180;

    // Draw immediately with generated pattern (real implementation
    // would decode audio and compute actual mel bins via Web Audio API)
    drawMelSpectrogram(canvas, null);
    setMelReady(true);
  }, [file]);

  const togglePlay = () => {
    wsRef.current?.playPause();
    setPlaying((p) => !p);
  };

  const fmt = (s) => s ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : "--:--";

  return (
    <div className="two-col">
      {/* Waveform + playback card */}
      <div className="card">
        <div className="card-title">Waveform Preview</div>
        <div className="player-wrap">
          <div ref={waveRef} style={{ width: "100%" }} />
          <div className="player-controls">
            <button className="play-btn" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
              {playing ? "⏸" : "▶"}
            </button>
            <div>
              <div className="player-filename">{file?.name}</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-3)", marginTop: 3 }}>
                {fmt(duration)} · {file ? (file.size / 1024).toFixed(0) + " KB" : ""}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mel spectrogram card */}
      <div className="card">
        <div className="card-title">Mel Spectrogram</div>
        <div className="mel-wrap">
          {file ? (
            <>
              <canvas ref={melRef} style={{ width: "100%", display: "block", borderRadius: 6 }} />
              {melReady && (
                <div style={{
                  position: "absolute", bottom: 6, right: 10,
                  fontFamily: "var(--font-mono)", fontSize: 9,
                  color: "rgba(255,255,255,0.4)"
                }}>
                  128 mel bins · 44.1 kHz
                </div>
              )}
            </>
          ) : (
            <div className="mel-placeholder">Mel spectrogram will appear here after upload</div>
          )}
        </div>
        {/* Colour scale legend */}
        {file && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-3)" }}>Low</span>
            <div style={{
              flex: 1, height: 6, borderRadius: 3,
              background: "linear-gradient(90deg, #150850, #2a5cb4, #40c8c0, #f0eb50)"
            }} />
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-3)" }}>High</span>
          </div>
        )}
      </div>
    </div>
  );
}