import os
import json
import numpy as np
import requests
import base64
import sys
from io import BytesIO
from PIL import Image
from datetime import datetime
import pathlib
import torch
import firebase_admin
from firebase_admin import credentials, firestore

from cloudinary import config as cloudinary_config
from cloudinary.api import resources
from fastapi import FastAPI, HTTPException
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath



# --- Setup YOLOv5 path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_DIR = os.path.join(BASE_DIR, 'yolov5')
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

# --- YOLOv5 dependencies ---
from yolov5.models.yolo import Model, DetectionModel
from yolov5.utils.general import non_max_suppression, scale_coords, check_yaml
from yolov5.utils.dataloaders import letterbox
from torch.serialization import safe_globals
from yolov5.utils.torch_utils import select_device

# --- Firebase initialization ---
firebase_base64 = os.getenv("FIREBASE_CREDENTIALS_JSON")
firebase_json_str = base64.b64decode(firebase_base64).decode('utf-8')
firebase_dict = json.loads(firebase_json_str)
cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(cred, {
    'projectId': 'bacs-view'
})
db = firestore.client()

# --- Cloudinary configuration ---
cloudinary_config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# --- FastAPI app ---
app = FastAPI()

# --- Load YOLOv5 model (using local working logic) ---
device = select_device('cpu')
cfg_path = check_yaml("yolov5/models/yolov5s.yaml")
model = Model(cfg_path, ch=3, nc=1).to(device)  # nc=1 for one class (car)

weights_path = "best(1).pt"  # Your model file
with safe_globals([DetectionModel]):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
state_dict = checkpoint['model'].state_dict()
model.load_state_dict(state_dict)
model.eval()

img_size = 640
stride = 64
pt = True

# --- Helper: fetch latest Cloudinary image URL ---
async def fetch_latest_image_url():
    res = resources(type='upload', max_results=1, direction='desc')
    if res['resources']:
        return res['resources'][0]['secure_url']
    return None

# --- Detection endpoint ---
@app.get("/detect")
async def detect_latest_image():
    image_url = await fetch_latest_image_url()
    if not image_url:
        raise HTTPException(status_code=404, detail="No images found in Cloudinary")

    try:
        # Download image from Cloudinary
        response = requests.get(image_url)
        img0 = Image.open(BytesIO(response.content)).convert("RGB")  # Original image

        # Preprocess (same as local code)
        img = letterbox(np.array(img0), new_shape=img_size, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))  # CHW
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)
            if isinstance(pred, tuple):  # if tuple output
                pred = pred[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # Post-processing
        car_count = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.size).round()
                car_count += (det[:, -1] == 0).sum().item()

        # Log detection to Firebase
        timestamp = datetime.utcnow().isoformat()
        db.collection("detections").document().set({
            "timestamp": timestamp,
            "image_url": image_url,
            "car_count": car_count
        })

        return {
            "status": "success",
            "timestamp": timestamp,
            "car_count": car_count,
            "image_url": image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
