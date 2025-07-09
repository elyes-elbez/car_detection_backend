
import os
import json
import asyncio
import requests
from io import BytesIO
from PIL import Image
import torch
import firebase_admin
from firebase_admin import credentials, firestore
from cloudinary import config as cloudinary_config
from cloudinary.api import resources
from fastapi import FastAPI
import os
from cloudinary import config as cloudinary_config
import base64

# Decode the base64 credential from the environment variable
firebase_base64 = os.getenv("FIREBASE_CREDENTIALS_JSON")
firebase_json = base64.b64decode(firebase_base64)

# Load credentials from the decoded content
cred = credentials.Certificate(BytesIO(firebase_json))
firebase_admin.initialize_app(cred, {
    'projectId': 'bacs-view'
})

# Get Firestore client
db = firestore.client()

# Configure Cloudinary
cloudinary_config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

app = FastAPI()

# Load your YOLOv5 custom model via Torch Hub (will download automatically on first run)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

async def fetch_latest_image_url():
    res = resources(type='upload', max_results=1, direction='desc')
    if res['resources']:
        return res['resources'][0]['secure_url']
    return None

async def detect_and_save():
    while True:
        image_url = await fetch_latest_image_url()
        if image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            results = model(img)

            # Count cars (class 0 usually = car)
            car_count = (results.pred[0][:, -1] == 0).sum().item()

            from datetime import datetime
            timestamp = datetime.utcnow().isoformat()

            # Save detection to Firestore collection 'detections'
            doc_ref = db.collection('detections').document()  # new auto-ID doc
            doc_ref.set({
                'timestamp': timestamp,
                'image_url': image_url,
                'car_count': car_count
            })

            print(f"[{timestamp}] Detected {car_count} cars.")

        await asyncio.sleep(600)  # wait 10 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detect_and_save())
