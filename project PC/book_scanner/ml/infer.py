"""
Inferensi deteksi dokumen dengan model YOLOv8-seg hasil training.

API utama:
    detect_corners(image_bgr) -> (corners_4x2_float | None, confidence_float)

Model dimuat sekali (lazy). Bila bobot/torch tidak tersedia, mengembalikan
None sehingga pemanggil bisa fallback ke metode klasik (GrabCut/kontur).
"""
import os
import threading

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'weights', 'book_seg.pt')

_model = None
_load_failed = False
_lock = threading.Lock()


def _order_points(pts):
    pts = np.asarray(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _get_model():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is None and not _load_failed:
            try:
                if not os.path.exists(WEIGHTS):
                    _load_failed = True
                    return None
                from ultralytics import YOLO
                _model = YOLO(WEIGHTS)
            except Exception as e:
                print(f"[infer] gagal memuat model: {e}")
                _load_failed = True
    return _model


def _mask_to_quad(mask):
    """Ubah mask biner menjadi 4 sudut (approxPolyDP -> minAreaRect)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2))
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return _order_points(box)


def detect_corners(image_bgr, conf=0.25, imgsz=512):
    """
    Kembalikan (corners 4x2 dalam koordinat piksel gambar asli, confidence).
    None bila model tak tersedia atau tak ada deteksi meyakinkan.
    """
    model = _get_model()
    if model is None:
        return None, 0.0

    h, w = image_bgr.shape[:2]
    try:
        res = model.predict(image_bgr, imgsz=imgsz, conf=conf,
                            verbose=False, device='cpu')
    except Exception as e:
        print(f"[infer] predict error: {e}")
        return None, 0.0

    if not res:
        return None, 0.0
    r = res[0]
    if r.masks is None or len(r.masks) == 0:
        return None, 0.0

    # pilih instance dengan confidence tertinggi
    confs = r.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    best_conf = float(confs[idx])

    mask = r.masks.data[idx].cpu().numpy()  # (mh, mw) 0..1
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    quad = _mask_to_quad(mask)
    if quad is None:
        return None, best_conf
    return quad, best_conf


def available():
    """True bila model siap dipakai."""
    return _get_model() is not None
