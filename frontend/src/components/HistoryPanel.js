import React from "react";
import { getDevice } from "../config/scenes";

export default function HistoryPanel({ history }) {
  return (
    <div className="card">
      <div className="card-title">Prediction History</div>
      {!history?.length ? (
        <div className="empty-state">
          No predictions yet.<br />Upload an audio file to begin.
        </div>
      ) : (
        history.map((h, i) => {
          const deviceInfo = getDevice(h.device);
          return (
            <div className="history-item" key={i}>
              <div>
                <div className="history-scene">{h.scene}</div>
                <div className="history-device">
                  {deviceInfo.label}
                  {deviceInfo.sub && (
                    <span style={{ color: "var(--text-muted)" }}> ({deviceInfo.sub})</span>
                  )}
                </div>
              </div>
              {h.confidence && (
                <div className="history-badge">{h.confidence}%</div>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}