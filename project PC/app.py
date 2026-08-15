"""
ScanBook - Aplikasi Pengolahan Citra
Deteksi tepi buku, efek scanner (perspektif), serta kontrol kontras & saturasi.
"""
import os
import io
import base64
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


# ----------------------------------------------------------------------------
# Util: konversi gambar <-> base64
# ----------------------------------------------------------------------------
def b64_to_cv2(data_url: str):
    """Decode data URL base64 menjadi gambar OpenCV (BGR)."""
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img) -> str:
    """Encode gambar OpenCV (BGR) menjadi data URL PNG base64."""
    ok, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('utf-8')
    return 'data:image/png;base64,' + b64


# ----------------------------------------------------------------------------
# Pipeline deteksi tepi & koreksi perspektif (efek scanner)
# ----------------------------------------------------------------------------
def order_points(pts):
    """Urutkan 4 titik: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # kiri-atas
    rect[2] = pts[np.argmax(s)]   # kanan-bawah
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # kanan-atas
    rect[3] = pts[np.argmax(diff)]  # kiri-bawah
    return rect


def four_point_transform(image, pts):
    """Koreksi perspektif berdasarkan 4 titik sudut."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    maxWidth = max(maxWidth, 1)
    maxHeight = max(maxHeight, 1)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document(image):
    """
    Deteksi tepi dokumen/buku menggunakan Canny + kontur.
    Mengembalikan (gambar_edges, kontur_dokumen|None).
    """
    ratio = image.shape[0] / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.2 * resized.shape[0] * resized.shape[1]:
            doc_cnt = approx
            break

    return edged, doc_cnt, ratio


def make_scan(image, doc_cnt, ratio):
    """Hasilkan citra ter-scan dengan koreksi perspektif jika tepi ditemukan."""
    if doc_cnt is not None:
        warped = four_point_transform(image, doc_cnt.reshape(4, 2) * ratio)
    else:
        warped = image.copy()
    return warped


def scanner_effect(image):
    """Tampilan dokumen ala scanner: grayscale + adaptive threshold halus."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 12)
    # haluskan agar tidak terlalu pecah
    thr = cv2.medianBlur(thr, 3)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
# Penyesuaian kontras & saturasi
# ----------------------------------------------------------------------------
def adjust_contrast_saturation(image, contrast=1.0, saturation=1.0, brightness=0.0):
    """
    contrast    : 0.5 - 3.0  (1.0 = normal)
    saturation  : 0.0 - 3.0  (1.0 = normal)
    brightness  : -100 - 100 (0 = normal)
    """
    img = image.astype(np.float32)

    # kontras + kecerahan
    img = img * contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)

    # saturasi via HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan():
    """Deteksi tepi + koreksi perspektif. Mengembalikan beberapa hasil."""
    data = request.get_json()
    img = b64_to_cv2(data['image'])
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    edged, doc_cnt, ratio = detect_document(img)

    # gambar overlay kontur tepi pada gambar asli (untuk visualisasi)
    overlay = img.copy()
    detected = doc_cnt is not None
    if detected:
        cv2.drawContours(overlay,
                         [(doc_cnt.reshape(4, 2) * ratio).astype(int)],
                         -1, (0, 220, 120), max(3, img.shape[1] // 250))

    scanned = make_scan(img, doc_cnt, ratio)
    scan_bw = scanner_effect(scanned)
    edges_vis = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    return jsonify({
        'detected': detected,
        'edges': cv2_to_b64(edges_vis),
        'overlay': cv2_to_b64(overlay),
        'scanned': cv2_to_b64(scanned),
        'scan_bw': cv2_to_b64(scan_bw),
    })


@app.route('/adjust', methods=['POST'])
def adjust():
    """Sesuaikan kontras, saturasi, kecerahan pada gambar ter-scan."""
    data = request.get_json()
    img = b64_to_cv2(data['image'])
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    contrast = float(data.get('contrast', 1.0))
    saturation = float(data.get('saturation', 1.0))
    brightness = float(data.get('brightness', 0.0))

    out = adjust_contrast_saturation(img, contrast, saturation, brightness)
    return jsonify({'result': cv2_to_b64(out)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
