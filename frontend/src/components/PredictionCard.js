import React from "react";

export default function PredictionCard({ scene, device }) {

  return (

    <div className="card">

      <h2>Prediction Result</h2>

      <div className="prediction">

        <div>
          <h4>Scene</h4>
          <div className="scene">
            {scene}
          </div>
        </div>

        <div>
          <h4>Device</h4>
          <div className="device">
            {device}
          </div>
        </div>

      </div>

    </div>

  );

}