// import React from "react";

// export default function HistoryPanel({history}){

// return(

// <div className="card">

// <h3>Prediction History</h3>

// <ul>

// {history.map((h,i)=>(
// <li key={i}>
// {h.scene} - {h.device}
// </li>
// ))}

// </ul>

// </div>

// );

// }


import React from "react";

export default function HistoryPanel({ history }) {
  return (
    <div className="card">
      <div className="card-title">Prediction History</div>
      {!history?.length ? (
        <div className="empty-state">
          No predictions yet.<br />Upload an audio file to begin.
        </div>
      ) : (
        history.map((h, i) => (
          <div className="history-item" key={i}>
            <div>
              <div className="history-scene">{h.scene}</div>
              <div className="history-device">{h.device}</div>
            </div>
            {h.confidence && (
              <div className="history-badge">{h.confidence}%</div>
            )}
          </div>
        ))
      )}
    </div>
  );
}