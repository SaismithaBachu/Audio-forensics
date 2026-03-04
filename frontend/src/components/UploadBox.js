import React from "react";

export default function UploadBox({onUpload}){

const handleFile=(e)=>{
const file=e.target.files[0];
if(file) onUpload(file);
};

return(

<div
className="upload-box"
onClick={()=>document.getElementById("audio-upload").click()}
>

<h2>Upload Audio</h2>
<p>Click here to upload an audio file</p>

<input
id="audio-upload"
type="file"
accept="audio/*"
hidden
onChange={handleFile}
/>

</div>

);

}