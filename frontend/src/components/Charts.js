// import React from "react";
// import {
// BarChart,
// Bar,
// XAxis,
// YAxis,
// Tooltip,
// ResponsiveContainer
// } from "recharts";

// export default function Charts({data}){

// return(

// <div className="card">

// <h3>Scene Confidence</h3>

// <ResponsiveContainer width="100%" height={300}>

// <BarChart data={data}>

// <XAxis dataKey="scene"/>
// <YAxis/>
// <Tooltip/>

// <Bar dataKey="value" fill="#4facfe"/>

// </BarChart>

// </ResponsiveContainer>

// </div>

// );

// }

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#fff", border: "1px solid var(--border)",
      borderRadius: 8, padding: "10px 14px",
      fontFamily: "var(--font-mono)", fontSize: 12, boxShadow: "var(--shadow)"
    }}>
      <div style={{ color: "var(--text-2)", marginBottom: 4, textTransform: "capitalize" }}>{label}</div>
      <div style={{ color: "var(--indigo)", fontWeight: 600 }}>
        {(payload[0].value * 100).toFixed(1)}%
      </div>
    </div>
  );
};

export default function Charts({ data }) {
  const maxVal = Math.max(...data.map(d => d.value));
  const sorted = [...data].sort((a, b) => b.value - a.value);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Bar chart */}
      <div className="card">
        <div className="card-title">Scene Probability Distribution</div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} barCategoryGap="28%">
            <XAxis
              dataKey="scene"
              tick={{ fill: "var(--text-3)", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={{ stroke: "var(--border)" }} tickLine={false}
            />
            <YAxis
              tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "var(--text-3)", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={false} tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--indigo-pale)" }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.value === maxVal ? "#3d2fa9" : "#c4b9f5"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scene breakdown indicators */}
      <div className="card">
        <div className="card-title">Scene Breakdown — Top Matches</div>
        <div className="indicators-grid">
          {sorted.slice(0, 6).map((item, i) => {
            const pct = (item.value * 100).toFixed(1);
            const tag = i === 0 ? "top" : i <= 2 ? "mid" : "low";
            const tagLabel = i === 0 ? "TOP MATCH" : i <= 2 ? "POSSIBLE" : "UNLIKELY";
            return (
              <div className="ind-item" key={i}>
                <div className="ind-label" style={{ textTransform: "capitalize" }}>{item.scene}</div>
                <div className="ind-value" style={{ color: i === 0 ? "var(--indigo)" : "var(--text)" }}>
                  {pct}%
                </div>
                <div className={`ind-tag ${tag}`}>{tagLabel}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}