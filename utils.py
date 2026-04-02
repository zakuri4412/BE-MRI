import numpy as np
import cv2

IMG_SIZE = 128

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0,-1))
    return img

def postprocess_mask(pred, threshold=0.5):
    mask = (pred[0,...,0] >= threshold).astype(np.uint8) * 255
    return mask

def postprocess_heatmap(pred):
    heatmap = (pred[0,...,0] * 255).astype(np.uint8)
    return heatmap
