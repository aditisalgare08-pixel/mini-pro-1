"""
Face Detection System - One File Project
---------------------------------------
Python + Flask + OpenCV + HTML + CSS + JavaScript in one app.py file.

VS Code commands:
    python -m venv .venv
    .venv\Scripts\Activate.ps1          # Windows PowerShell
    pip install -r requirements.txt
    python app.py

Open in browser:
    http://127.0.0.1:5000

Features:
    1. Live webcam face detection
    2. Image upload face detection
    3. Face count report
    4. Modern responsive dashboard
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request


APP_NAME = "Face Detection System"
HOST = "127.0.0.1"
PORT = 5000
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class FaceDetector:
    def __init__(self) -> None:
        cascade = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.model = cv2.CascadeClassifier(cascade)
        if self.model.empty():
            raise RuntimeError("OpenCV face cascade not found.")

    def detect(self, img_bgr: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(45, 45),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


app = Flask(__name__)
detector = FaceDetector()


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ app_name }}</title>
<style>
:root{
    --bg:#0f172a;--card:#111827;--card2:#1f2937;--txt:#e5e7eb;
    --muted:#9ca3af;--blue:#38bdf8;--green:#22c55e;--red:#ef4444;
    --line:rgba(255,255,255,.12);--shadow:0 18px 55px rgba(0,0,0,.35)
}
*{box-sizing:border-box}
body{
    margin:0;min-height:100vh;font-family:Arial,Helvetica,sans-serif;color:var(--txt);
    background:radial-gradient(circle at top left,rgba(56,189,248,.22),transparent 35%),
    radial-gradient(circle at bottom right,rgba(34,197,94,.17),transparent 35%),var(--bg)
}
.wrap{width:min(1160px,calc(100% - 28px));margin:auto;padding:26px 0 38px}
.hero,.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px;margin-bottom:20px}
.card,.panel{
    background:rgba(17,24,39,.9);border:1px solid var(--line);border-radius:24px;
    box-shadow:var(--shadow);backdrop-filter:blur(14px)
}
.card{padding:28px;position:relative;overflow:hidden}
.card:after{content:"";position:absolute;right:-90px;top:-90px;width:230px;height:230px;
    border-radius:50%;background:rgba(56,189,248,.13)}
.badge{
    display:inline-flex;padding:8px 12px;border:1px solid rgba(56,189,248,.35);
    border-radius:999px;background:rgba(56,189,248,.08);color:#bae6fd;font-size:13px
}
h1{margin:14px 0 10px;font-size:clamp(32px,5vw,54px);line-height:1.04;letter-spacing:-1px}
p{color:var(--muted);line-height:1.65}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}
.stat,.mini{padding:15px;border-radius:18px;background:rgba(31,41,55,.75);border:1px solid var(--line)}
.stat b,.mini b{display:block;font-size:28px;margin-bottom:5px}
.stat span,.mini span{color:var(--muted);font-size:13px}
.steps{display:grid;gap:10px;margin:15px 0 0;padding:0;list-style:none}
.steps li{padding:12px 14px;border-radius:14px;background:rgba(31,41,55,.7);color:var(--muted)}
.grid{grid-template-columns:minmax(0,1fr) 360px;align-items:start}
.panel{padding:18px}
.toolbar{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:15px}
.row{display:flex;gap:10px;flex-wrap:wrap}
button,.filebtn{
    border:0;border-radius:14px;padding:11px 14px;background:var(--blue);font-weight:700;
    color:#06111f;cursor:pointer;transition:.15s;display:inline-flex;gap:8px;align-items:center
}
button:hover,.filebtn:hover{transform:translateY(-1px)}
button:disabled{opacity:.45;cursor:not-allowed;transform:none}
button.gray{background:#e5e7eb}.red{background:var(--red);color:white}.green{background:var(--green)}
input[type=file]{display:none}
.stage{
    position:relative;width:100%;aspect-ratio:16/10;border-radius:22px;overflow:hidden;
    background:#020617;border:1px solid var(--line)
}
video,canvas,img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
video{transform:scaleX(-1)}
canvas{pointer-events:none}
#photo{display:none;background:#020617}
.empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:var(--muted);padding:25px}
.empty b{display:block;color:var(--txt);font-size:20px;margin-bottom:8px}
.status{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:14px;color:var(--muted)}
.pill{display:inline-flex;gap:8px;align-items:center;padding:8px 11px;border-radius:999px;
    background:rgba(31,41,55,.75);border:1px solid var(--line);font-size:14px}
.dot{width:9px;height:9px;border-radius:50%;background:#f59e0b}.dot.on{background:var(--green);box-shadow:0 0 18px var(--green)}
h2{font-size:20px;margin:0 0 14px}
.mini{margin-bottom:12px}.mini label{display:block;color:var(--muted);font-size:13px;margin-bottom:8px}
.log{
    min-height:175px;max-height:255px;overflow:auto;padding:12px;border-radius:18px;
    background:rgba(2,6,23,.65);border:1px solid var(--line);font:13px/1.6 Consolas,monospace;color:#cbd5e1
}
.note{font-size:13px;margin-top:12px}.foot{text-align:center;color:var(--muted);font-size:13px;margin-top:20px}
@media(max-width:900px){.hero,.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="wrap">
<section class="hero">
    <div class="card">
        <div class="badge">Python + OpenCV + Flask + JavaScript</div>
        <h1>Face Detection System</h1>
        <p>Live webcam and image upload face detection. This project is simple for college practicals and easy to run in VS Code.</p>
        <div class="stats">
            <div class="stat"><b id="topFaces">0</b><span>Faces detected</span></div>
            <div class="stat"><b id="topMode">Idle</b><span>Current mode</span></div>
            <div class="stat"><b id="topFrames">0</b><span>Frames checked</span></div>
        </div>
    </div>
    <div class="card">
        <h2>Run Commands</h2>
        <ul class="steps">
            <li><b>1.</b> Open folder in VS Code</li>
            <li><b>2.</b> python -m venv .venv</li>
            <li><b>3.</b> .venv\Scripts\Activate.ps1</li>
            <li><b>4.</b> pip install -r requirements.txt</li>
            <li><b>5.</b> python app.py</li>
        </ul>
    </div>
</section>

<section class="grid">
    <div class="panel">
        <div class="toolbar">
            <div class="row">
                <button id="start" class="green">▶ Start Camera</button>
                <button id="stop" class="red" disabled>■ Stop</button>
                <button id="snap" class="gray" disabled>📸 Snapshot</button>
            </div>
            <div class="row">
                <label class="filebtn" for="file">🖼 Upload Image</label>
                <input id="file" type="file" accept="image/*">
                <button id="clear" class="gray">Clear</button>
            </div>
        </div>
        <div class="stage">
            <video id="video" autoplay muted playsinline></video>
            <img id="photo" alt="preview">
            <canvas id="canvas"></canvas>
            <div class="empty" id="empty"><div><b>Ready</b>Start camera or upload image.</div></div>
        </div>
        <div class="status">
            <span class="pill"><span class="dot" id="dot"></span><span id="msg">System idle</span></span>
            <span class="pill" id="backend">Backend checking...</span>
        </div>
    </div>

    <aside class="panel">
        <h2>Detection Report</h2>
        <div class="mini"><label>Detected Faces</label><b id="faces">0</b></div>
        <div class="mini"><label>Mode</label><b id="mode">Idle</b></div>
        <div class="mini"><label>Last Detection</label><b id="time">--</b></div>
        <h2>Activity Log</h2>
        <div class="log" id="log"></div>
        <p class="note">Tip: Good lighting and front-facing face gives better results.</p>
    </aside>
</section>
<div class="foot">Created using HTML, CSS, JavaScript and Python Flask.</div>
</main>

<script>
const $ = id => document.getElementById(id);
const video=$("video"), photo=$("photo"), canvas=$("canvas"), ctx=canvas.getContext("2d");
const start=$("start"), stop=$("stop"), snap=$("snap"), file=$("file"), clear=$("clear");
const empty=$("empty"), dot=$("dot"), msg=$("msg"), backend=$("backend");
const faces=$("faces"), mode=$("mode"), last=$("time"), logBox=$("log");
const topFaces=$("topFaces"), topMode=$("topMode"), topFrames=$("topFrames");
let stream=null,timer=null,frames=0,busy=false,currentMode="Idle";

function log(t){logBox.innerHTML=`[${new Date().toLocaleTimeString()}] ${t}<br>`+logBox.innerHTML}
function setMode(m){currentMode=m;mode.textContent=m;topMode.textContent=m}
function setFaces(n){faces.textContent=n;topFaces.textContent=n}
function setMsg(t,on=false){msg.textContent=t;dot.classList.toggle("on",on)}
function controls(on){start.disabled=on;stop.disabled=!on;snap.disabled=!on}
function fitCanvas(){let r=canvas.parentElement.getBoundingClientRect();canvas.width=r.width;canvas.height=r.height}
function clearCanvas(){fitCanvas();ctx.clearRect(0,0,canvas.width,canvas.height)}
function show(type){
    video.style.display=type==="video"?"block":"none";
    photo.style.display=type==="photo"?"block":"none";
    empty.style.display=type==="empty"?"grid":"none";
}
function drawBoxes(boxes,sw,sh,mirror=false){
    clearCanvas();
    let cw=canvas.width,ch=canvas.height,sr=sw/sh,cr=cw/ch,dw=cw,dh=ch,ox=0,oy=0;
    if(sr>cr){dh=cw/sr;oy=(ch-dh)/2}else{dw=ch*sr;ox=(cw-dw)/2}
    ctx.lineWidth=4;ctx.font="bold 16px Arial";ctx.textBaseline="top";
    boxes.forEach((b,i)=>{
        let x=ox+b.x/sw*dw,y=oy+b.y/sh*dh,w=b.w/sw*dw,h=b.h/sh*dh;
        if(mirror)x=cw-x-w;
        ctx.strokeStyle="#22c55e";ctx.fillStyle="rgba(34,197,94,.15)";
        ctx.strokeRect(x,y,w,h);ctx.fillRect(x,y,w,h);
        ctx.fillStyle="#22c55e";ctx.fillRect(x,Math.max(0,y-28),88,26);
        ctx.fillStyle="#04130a";ctx.fillText(`Face ${i+1}`,x+8,Math.max(0,y-24));
    });
}
function videoFrame(){
    if(!video.videoWidth)return null;
    let c=document.createElement("canvas");c.width=video.videoWidth;c.height=video.videoHeight;
    c.getContext("2d").drawImage(video,0,0,c.width,c.height);
    return c.toDataURL("image/jpeg",.82);
}
async function detect(data,sw,sh,mirror=false){
    if(busy)return;busy=true;
    try{
        let res=await fetch("/detect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image:data})});
        let out=await res.json();
        if(!out.ok)throw new Error(out.error||"Detection failed");
        frames++;topFrames.textContent=frames;setFaces(out.count);last.textContent=out.time;
        drawBoxes(out.faces,sw,sh,mirror);
        setMsg(out.count?`${out.count} face detected`:"No face detected",currentMode==="Live");
    }catch(e){setMsg("Detection error");log("Error: "+e.message)}
    finally{busy=false}
}
async function startCamera(){
    try{
        stopCamera();
        stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
        video.srcObject=stream;await video.play();show("video");setMode("Live");setMsg("Live camera running",true);
        controls(true);log("Camera started");
        timer=setInterval(()=>{let data=videoFrame();if(data)detect(data,video.videoWidth,video.videoHeight,true)},650);
    }catch(e){setMsg("Camera not available");log("Camera permission blocked");controls(false)}
}
function stopCamera(){
    if(timer){clearInterval(timer);timer=null}
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}
    video.srcObject=null;controls(false);
    if(currentMode==="Live"){setMode("Idle");setMsg("Camera stopped");log("Camera stopped")}
}
async function snapshot(){
    let data=videoFrame();if(!data)return log("Snapshot failed");
    setMode("Snapshot");setMsg("Snapshot detection running");log("Snapshot captured");
    await detect(data,video.videoWidth,video.videoHeight,true);
}
function readFile(f){return new Promise((ok,bad)=>{let r=new FileReader();r.onload=()=>ok(r.result);r.onerror=()=>bad();r.readAsDataURL(f)})}
async function upload(e){
    let f=e.target.files[0];if(!f)return;
    if(!f.type.startsWith("image/"))return log("Select valid image file");
    stopCamera();let data=await readFile(f);
    photo.onload=()=>{show("photo");setMode("Upload");setMsg("Image detection running");log("Image uploaded: "+f.name);detect(data,photo.naturalWidth,photo.naturalHeight,false)};
    photo.src=data;
}
function reset(){
    stopCamera();clearCanvas();show("empty");setMode("Idle");setFaces(0);setMsg("System idle");last.textContent="--";file.value="";log("Screen cleared");
}
async function health(){
    try{let r=await fetch("/health");let j=await r.json();backend.textContent=j.ok?"Backend online":"Backend issue"}
    catch(e){backend.textContent="Backend offline"}
}
start.onclick=startCamera;stop.onclick=stopCamera;snap.onclick=snapshot;file.onchange=upload;clear.onclick=reset;
window.onresize=clearCanvas;fitCanvas();show("empty");health();log("Application loaded");
</script>
</body>
</html>
"""


def data_url_to_bytes(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image data.")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Only image files are allowed.")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image too large. Use image below 8 MB.")
    return raw


def image_bytes_to_bgr(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed.")
    return img


@app.route("/")
def index():
    return render_template_string(PAGE, app_name=APP_NAME)


@app.route("/health")
def health():
    return jsonify({"ok": True, "app": APP_NAME, "opencv": cv2.__version__})


@app.route("/detect", methods=["POST"])
def detect():
    try:
        payload = request.get_json(silent=True) or {}
        raw = data_url_to_bytes(payload.get("image", ""))
        img = image_bytes_to_bgr(raw)
        found = detector.detect(img)
        return jsonify({
            "ok": True,
            "count": len(found),
            "faces": [face.to_dict() for face in found],
            "time": time.strftime("%H:%M:%S"),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"ok": False, "error": "Page not found"}), 404


if __name__ == "__main__":
    print("=" * 55)
    print(APP_NAME)
    print(f"Open: http://{HOST}:{PORT}")
    print("Stop server: CTRL + C")
    print("=" * 55)
    app.run(host=HOST, port=PORT, debug=True)
