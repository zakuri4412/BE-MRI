import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import cv2
from PIL import Image
import numpy as np

from model import model
from utils import preprocess, postprocess_mask, postprocess_heatmap
from fastapi.middleware.cors import CORSMiddleware
import nibabel as nib
import tempfile
import time
from fastapi import Form
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
VOLUME_CACHE = {}
def compute_metrics(mask, pred, inference_time):
    tumor_pixels = int(np.sum(mask > 0))
    total_pixels = mask.size
    tumor_ratio = float(tumor_pixels / total_pixels)

    prob_map = pred[0, ..., 0]

    # probability stats
    mean_prob = float(np.mean(prob_map))
    max_prob = float(np.max(prob_map))

    # bounding box
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    else:
        bbox = None

    return {
        "tumor_pixels": tumor_pixels,
        "tumor_ratio": tumor_ratio,
        "mean_probability": mean_prob,
        "max_probability": max_prob,
        "bbox": bbox,
        "inference_time_ms": round(inference_time * 1000, 2),
    }
@app.post("/predict-mask")
async def predict_mask(file: UploadFile = File(...), threshold: float = Form(0.5)):

    contents = await file.read()
    img = preprocess(contents)

    pred = model.predict(img)

    mask = postprocess_mask(pred, threshold)

    pil_img = Image.fromarray(mask)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    start = time.time()
    pred = model.predict(img)
    inference_time = time.time() - start
    metrics = compute_metrics(mask,pred,inference_time)

    _, mask_png = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(mask_png).decode()
    
    return {
        "mask": mask_b64,
        "metrics": metrics
    }
    # return StreamingResponse(buf, media_type="image/png")


@app.post("/predict-heatmap")
async def predict_heatmap(file: UploadFile = File(...)):

    contents = await file.read()
    img = preprocess(contents)

    pred = model.predict(img)

    heatmap = postprocess_heatmap(pred)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    pil_img = Image.fromarray(heatmap_color)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/predict-prob")
async def predict_prob(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess(contents)

    pred = model.predict(img)

    prob = (pred[0, ..., 0] * 255).astype(np.uint8)

    _, png = cv2.imencode(".png", prob)
    b64 = base64.b64encode(png).decode()

    return {"prob": b64}


@app.post("/predict-volume-slice")
async def predict_volume_slice(
    volume_id: str = Form(...),
    slice_index: int = Form(...)
):
    # ===== lấy volume từ cache =====
    volume = VOLUME_CACHE.get(volume_id)

    if volume is None:
        return {"error": "Volume not found"}

    # clamp index
    slice_index = int(slice_index)
    slice_index = max(0, min(slice_index, volume.shape[2] - 1))

    # ===== lấy slice =====
    slice_img = volume[:, :, slice_index]

    # resize như lúc train
    slice_img = cv2.resize(slice_img, (128, 128))
    input_img = slice_img[np.newaxis, ..., np.newaxis]

    # ===== predict =====
    pred = model.predict(input_img)[0, ..., 0]

    mask = (pred >= 0.5).astype(np.uint8) * 255

    # encode
    _, img_png = cv2.imencode(".png", (slice_img * 255).astype(np.uint8))
    _, mask_png = cv2.imencode(".png", mask)

    return {
        "slice": base64.b64encode(img_png).decode(),
        "mask": base64.b64encode(mask_png).decode(),
        "num_slices": volume.shape[2]
    }
    

@app.post("/upload-volume")
async def upload_volume(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif filename.endswith(".nii"):
        suffix = ".nii"
    else:
        suffix = ".nii.gz"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    volume_id = str(uuid.uuid4())
    VOLUME_CACHE[volume_id] = volume

    return {
        "volume_id": volume_id,
        "num_slices": volume.shape[2]
    }