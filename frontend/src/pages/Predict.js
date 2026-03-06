// import React, { useState } from "react";

// import UploadBox from "../components/UploadBox";
// import PredictionCard from "../components/PredictionCard";
// import Charts from "../components/Charts";
// import AudioPlayer from "../components/AudioPlayer";
// import Loader from "../components/Loader";

// import { predictAudio } from "../api/api";
// import { SCENES } from "../config/scenes";

// export default function Predict() {

//   const [file, setFile] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleUpload = async (f) => {

//     // reset previous result
//     setResult(null);
//     setFile(f);
//     setLoading(true);

//     try {

//       const res = await predictAudio(f);
//       setResult(res);

//     } catch (err) {
//       console.error(err);
//       alert("Prediction failed");
//     }

//     setLoading(false);
//   };

//   let chartData = [];

//   if (result && result.scene_probs) {

//     chartData = SCENES.map((scene, i) => ({
//       scene,
//       value: result.scene_probs[i] || 0
//     }));

//   }

//   return (

//     <div className="analytics-container">

//       {/* Upload area */}
//       <UploadBox onUpload={handleUpload} />

//       {/* Audio preview */}
//       {file && (
//         <AudioPlayer file={file} />
//       )}

//       {/* Loading */}
//       {loading && <Loader />}

//       {/* Results */}
//       {result && !loading && (

//         <>
//           <PredictionCard
//             scene={result.scene}
//             device={result.device}
//           />

//           {chartData.length > 0 && (
//             <Charts data={chartData} />
//           )}
//         </>

//       )}

//     </div>

//   );

// }

import React, { useState } from "react";
import UploadBox from "../components/UploadBox";
import PredictionCard from "../components/PredictionCard";
import Charts from "../components/Charts";
import AudioPlayer from "../components/AudioPlayer";
import Loader from "../components/Loader";
import HistoryPanel from "../components/HistoryPanel";
import { predictAudio } from "../api/api";
import { SCENES } from "../config/scenes";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleUpload = async (f) => {
    setResult(null);
    setFile(f);
    setLoading(true);
    try {
      const res = await predictAudio(f);
      setResult(res);
      const conf = res.scene_probs ? Math.round(Math.max(...res.scene_probs) * 100) : null;
      setHistory(prev => [{ scene: res.scene, device: res.device, confidence: conf }, ...prev.slice(0, 9)]);
    } catch (err) {
      console.error(err);
      alert("Prediction failed — make sure the backend is running on port 8000.");
    }
    setLoading(false);
  };

  const chartData = result?.scene_probs
    ? SCENES.map((scene, i) => ({ scene, value: result.scene_probs[i] || 0 }))
    : [];

  return (
    <div className="analysis-page">
      <div className="page-header">
        <div className="page-tag">Analysis Dashboard</div>
        <h1 className="page-title">
          {result
            ? <span style={{ textTransform: "capitalize" }}>{result.scene}</span>
            : "Upload Audio"}
        </h1>
        <div className="page-sub">
          {file
            ? `${file.name}${result ? ` · ${result.device} detected` : " · Processing..."}`
            : "Upload an audio file to begin acoustic scene analysis"}
        </div>
      </div>

      <div className="analysis-grid">
        {/* Main column */}
        <div className="main-col">

          {/* Upload card — full width */}
          <div className="card">
            <div className="card-title">Upload Audio</div>
            <UploadBox onUpload={handleUpload} />
          </div>

          {/* Empty state placeholders before upload */}
          {!file && (
            <div className="two-col">
              <div className="card">
                <div className="card-title">Waveform Preview</div>
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "center",
                  height: 120, fontFamily: "var(--font-mono)", fontSize: 13,
                  color: "var(--text-3)"
                }}>
                  Awaiting upload...
                </div>
              </div>
              <div className="card">
                <div className="card-title">Mel Spectrogram</div>
                <div className="mel-wrap">
                  <div className="mel-placeholder">Mel spectrogram will appear here after upload</div>
                </div>
              </div>
            </div>
          )}

          {/* Waveform + mel after upload — two cards side by side */}
          {file && <AudioPlayer file={file} />}

          {/* Loader */}
          {loading && <Loader />}

          {/* Results */}
          {result && !loading && (
            <>
              <PredictionCard
                scene={result.scene}
                device={result.device}
                sceneProbs={result.scene_probs}
                deviceProbs={result.device_probs}
                filename={file?.name}
              />
              {chartData.length > 0 && <Charts data={chartData} />}
            </>
          )}
        </div>

        {/* Sidebar */}
        <div className="side-col">
          <HistoryPanel history={history} />
        </div>
      </div>
    </div>
  );
}