import React from "react";

export default function HistoryPanel({history}){

return(

<div className="card">

<h3>Prediction History</h3>

<ul>

{history.map((h,i)=>(
<li key={i}>
{h.scene} - {h.device}
</li>
))}

</ul>

</div>

);

}