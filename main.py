import os
import json
import asyncio
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from cloudinary import config as cloudinary_config
from cloudinary.api import resources
from fastapi import FastAPI
import base64
from datetime import datetime

# --- Firebase initialization with base64 env var ---
firebase_base64 = os.getenv("FIREBASE_CREDENTIALS_JSON")
firebase_json_str = base64.b64decode(firebase_base64).decode('utf-8')
firebase_dict = json.loads(firebase_json_str)
cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(cred, {
    'projectId': 'bacs-view'
})
db = firestore.client()

# --- Cloudinary config ---
cloudinary_config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

app = FastAPI()

# --- Load your trained YOLO model ---
model = YOLO('best(1).pt')  # make sure best(1).pt is in your repo root or specify path

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

            results = model(img)  # Run inference

            detections = results[0].boxes  # Boxes object
            # Count cars (class 0 is car)
            car_count = sum(box.cls.item() == 0 for box in detections)

            timestamp = datetime.utcnow().isoformat()

            # Save detection to Firestore collection 'detections'
            doc_ref = db.collection('detections').document()
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
