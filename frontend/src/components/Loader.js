import React from "react";

export default function Loader() {
  return (
    <div className="card">
      <div className="loader-wrap">
        <div className="loader-ring" />
        <div className="loader-text">Analysing audio...</div>
        <div className="loader-sub">Computing mel-spectrogram · Running CRNN model</div>
      </div>
    </div>
  );
}