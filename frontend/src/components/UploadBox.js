import React, { useState } from "react";

export default function UploadBox({ onUpload }) {
  const [dragging, setDragging] = useState(false);

  const handle = (file) => { if (file) onUpload(file); };

  return (
    <div
      className={`upload-zone${dragging ? " dragover" : ""}`}
      onClick={() => document.getElementById("audio-upload").click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]); }}
    >
      <div className="upload-icon">🎵</div>
      <h3>Upload Audio File</h3>
      <p>Drag & drop or click to browse — WAV, MP3, FLAC supported</p>
      <div className="btn-primary" style={{ pointerEvents: "none" }}>Browse Files</div>
      <input id="audio-upload" type="file" accept="audio/*" hidden
        onChange={(e) => handle(e.target.files[0])} />
    </div>
  );
}