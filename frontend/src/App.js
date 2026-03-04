import React from "react";
import {BrowserRouter,Routes,Route,Link} from "react-router-dom";

import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Analytics from "./pages/Analytics";

import "./styles.css";

export default function App(){

return(

<BrowserRouter>

<div className="navbar">

<h2>Audio Forensics AI</h2>

<div className="nav-links">

<Link to="/">Home</Link>
<Link to="/predict">Predict</Link>
<Link to="/analytics">Analytics</Link>

</div>

</div>

<Routes>

<Route path="/" element={<Home/>}/>
<Route path="/predict" element={<Predict/>}/>
<Route path="/analytics" element={<Analytics/>}/>

</Routes>

</BrowserRouter>

);

}