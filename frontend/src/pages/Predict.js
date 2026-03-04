import React, { useState } from "react";

import UploadBox from "../components/UploadBox";
import PredictionCard from "../components/PredictionCard";
import Charts from "../components/Charts";
import AudioPlayer from "../components/AudioPlayer";
import Loader from "../components/Loader";

import { predictAudio } from "../api/api";
import { SCENES } from "../config/scenes";

export default function Predict() {

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (f) => {

    // reset previous result
    setResult(null);
    setFile(f);
    setLoading(true);

    try {

      const res = await predictAudio(f);
      setResult(res);

    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  let chartData = [];

  if (result && result.scene_probs) {

    chartData = SCENES.map((scene, i) => ({
      scene,
      value: result.scene_probs[i] || 0
    }));

  }

  return (

    <div className="analytics-container">

      {/* Upload area */}
      <UploadBox onUpload={handleUpload} />

      {/* Audio preview */}
      {file && (
        <AudioPlayer file={file} />
      )}

      {/* Loading */}
      {loading && <Loader />}

      {/* Results */}
      {result && !loading && (

        <>
          <PredictionCard
            scene={result.scene}
            device={result.device}
          />

          {chartData.length > 0 && (
            <Charts data={chartData} />
          )}
        </>

      )}

    </div>

  );

}