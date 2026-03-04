import React,{useEffect, useRef} from "react";
import WaveSurfer from "wavesurfer.js";

export default function AudioPlayer({file}){

const ref=useRef(null);

useEffect(()=>{

if(!file) return;

const wavesurfer=WaveSurfer.create({

container:ref.current,
waveColor:"#4facfe",
progressColor:"#ff4d4d"

});

wavesurfer.loadBlob(file);

return ()=>wavesurfer.destroy();

},[file]);

return(

<div className="card">

<h3>Audio Waveform</h3>

<div ref={ref}></div>

</div>

);

}