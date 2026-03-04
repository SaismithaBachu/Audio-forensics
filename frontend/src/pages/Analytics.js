import React from "react";
import Charts from "../components/Charts";

export default function Analytics(){

const dummy=[
{scene:"airport",value:3},
{scene:"bus",value:2},
{scene:"park",value:4}
];

return(

<div className="analytics-container">

<h1>Visual Analytics</h1>

<Charts data={dummy}/>

</div>

);

}