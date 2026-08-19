import numpy as np
import io
import os
import logging

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("garbage-classifier")

CHECKPOINT_PATH = os.getenv("MODEL_PATH", "garbage_cnn.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_FILE_SIZE_MB = 10

app = FastAPI(
    title="Garbage Classification API",
    description="Upload an image and get the predicted waste category from a from-scratch CNN.",
    version="1.0.0",
)

model, checkpoint = load_model(CHECKPOINT_PATH, device=DEVICE)
CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE = checkpoint.get("img_size", 128)
MEAN = checkpoint.get("mean", [0.485, 0.456, 0.406])
STD = checkpoint.get("std", [0.229, 0.224, 0.225])

# ---- Base eval transform + TTA variants ----
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_transforms = [
    eval_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAutocontrast(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]

logger.info(f"Model loaded on {DEVICE}. Classes: {CLASS_NAMES}. TTA views: {len(tta_transforms)}")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("static", "index.html")) as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "num_classes": len(CLASS_NAMES)}


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}


def predict_with_tta(image: Image.Image) -> np.ndarray:
    probs_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(image).unsqueeze(0).to(DEVICE)
            probs = F.softmax(model(tensor), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(tta_transforms)).cpu().numpy().flatten()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    probs = predict_with_tta(image)

    top_idx = int(probs.argmax())
    result = {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
        },
    }
    return JSONResponse(result)
