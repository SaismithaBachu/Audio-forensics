import React from "react";
import {
BarChart,
Bar,
XAxis,
YAxis,
Tooltip,
ResponsiveContainer
} from "recharts";

export default function Charts({data}){

return(

<div className="card">

<h3>Scene Confidence</h3>

<ResponsiveContainer width="100%" height={300}>

<BarChart data={data}>

<XAxis dataKey="scene"/>
<YAxis/>
<Tooltip/>

<Bar dataKey="value" fill="#4facfe"/>

</BarChart>

</ResponsiveContainer>

</div>

);

}