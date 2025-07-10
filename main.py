import os
import json
import requests
import base64
import sys
from io import BytesIO
from PIL import Image
from datetime import datetime

import torch
import firebase_admin
from firebase_admin import credentials, firestore

from cloudinary import config as cloudinary_config
from cloudinary.api import resources
from fastapi import FastAPI, HTTPException

# --- Setup YOLOv5 path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_DIR = os.path.join(BASE_DIR, 'yolov5')

if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

# --- Import YOLOv5 dependencies ---
from models.yolo import Model
from utils.general import check_yaml, non_max_suppression, scale_coords
from utils.dataloaders import letterbox
from utils.torch_utils import select_device

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

# --- Load YOLOv5s model ---
device = select_device("cpu")
cfg_path = check_yaml("yolov5/models/yolov5s.yaml")
model = Model(cfg_path, ch=3, nc=1).to(device)  # nc=1 for single class

weights_path = "best_weights_only.pt"
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

stride = 32
names = ['car']
pt = True
img_size = 320  # reduce to 320 if memory is tight

# --- Helper to fetch latest image URL from Cloudinary ---
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
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        img0 = img.copy()
        img = letterbox(img0, new_shape=img_size, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        car_count = 0
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.size).round()
                car_count += (det[:, -1] == 0).sum().item()

        timestamp = datetime.utcnow().isoformat()

        # Save detection result to Firebase
        db.collection("detections").document().set({
            "timestamp": timestamp,
            "image_url": image_url,
            "car_count": car_count
        })

        # Free memory
        del img, img0, pred, det
        torch.cuda.empty_cache()

        return {
            "status": "success",
            "timestamp": timestamp,
            "car_count": car_count,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
