# Face Detection System

A simple VS Code-ready Face Detection System using:

- Python
- Flask
- OpenCV
- HTML
- CSS
- JavaScript

## Run in VS Code

Open this folder in VS Code, then open Terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## If PowerShell activation error comes

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

## Features

- Live webcam face detection
- Image upload face detection
- Face count dashboard
- Modern UI
- One main code file: `app.py`

## Packages

```text
flask
opencv-python
numpy
```
