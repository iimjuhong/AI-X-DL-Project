import os
import shutil
import cv2
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load Model
model = None
if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please place 'best.pt' in the backend directory.")
    # Fallback to a standard model for demonstration if best.pt is missing, 
    # but strictly we should wait for the user's model.
    # model = YOLO("yolov8n.pt") 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model 'best.pt' not found on server.")

    try:
        # Save uploaded file
        file_location = UPLOAD_DIR / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Inference
        results = model(str(file_location), imgsz=640, conf=0.25)
        
        # Get the plotted image (numpy array, BGR)
        plotted_image = results[0].plot()
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', plotted_image)
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        
        # Get detection info (boxes)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": results[0].names[int(box.cls)],
                    "conf": float(box.conf),
                    "xywh": box.xywh.tolist()
                })

        return JSONResponse(content={
            "message": "Success",
            "image": f"data:image/jpeg;base64,{encoded_string}",
            "detections": detections
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Serve Frontend
# We assume frontend files are in ../frontend relative to this file
FRONTEND_DIR = BASE_DIR.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
