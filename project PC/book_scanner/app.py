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
# User-facing upload cap is 16 MB (raw file), enforced client-side. The request
# body is a base64 data URL inside JSON (~1.35x), so allow headroom here.
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024  # 24 MB


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


def _quad_score(quad, area_img):
    """
    Nilai kelayakan sebuah quad sebagai dokumen.
    Lebih tinggi = lebih bagus. Menggabungkan luas relatif & kemiripan
    persegi panjang (sudut mendekati 90 derajat, sisi seimbang).
    """
    quad = quad.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(quad)
    if area < 0.10 * area_img:
        return -1.0

    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    # rasio sisi atas/bawah & kiri/kanan harus mirip
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    if min(top, bottom, left, right) < 1:
        return -1.0
    hbal = min(top, bottom) / max(top, bottom)
    vbal = min(left, right) / max(left, right)

    # sudut mendekati 90 derajat
    def angle(a, b, c):
        v1, v2 = a - b, c - b
        cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    angs = [angle(bl, tl, tr), angle(tl, tr, br),
            angle(tr, br, bl), angle(br, bl, tl)]
    ang_pen = np.mean([abs(a - 90) for a in angs]) / 90.0

    area_ratio = area / area_img
    return area_ratio + 0.6 * hbal + 0.6 * vbal - 0.8 * ang_pen


def _candidate_edges(gray):
    """Hasilkan beberapa peta tepi dari strategi berbeda."""
    maps = []
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # 1) Canny adaptif berbasis median
    med = np.median(blur)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    maps.append(cv2.Canny(blur, lo, hi))

    # 2) Canny kontras tinggi
    maps.append(cv2.Canny(blur, 50, 150))

    # 3) Morphological gradient (tangguh untuk tepi lembut/bayangan)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k)
    _, grad = cv2.threshold(grad, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps.append(grad)

    return maps


def segment_foreground(resized):
    """
    Pisahkan objek utama (dokumen) dari latar memakai GrabCut.
    Mengembalikan mask biner (255=foreground) atau None bila gagal/terlalu kecil.
    """
    h, w = resized.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx, my = int(0.04 * w), int(0.04 * h)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    try:
        cv2.grabCut(resized, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    # ambil hanya komponen foreground terbesar (buang bercak kecil)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.12 * h * w:
        return None
    clean = np.zeros_like(fg)
    cv2.drawContours(clean, [biggest], -1, 255, cv2.FILLED)
    return clean


def _try_ml_detector(image):
    """Coba detektor YOLOv8-seg; (quad_4x2_koordinat_asli, conf) atau (None, 0)."""
    try:
        from ml import infer as ml_infer
        return ml_infer.detect_corners(image)
    except Exception as e:
        print(f"[ml] detektor tidak aktif: {e}")
        return None, 0.0


def detect_document(image):
    """
    Deteksi dokumen. Urutan: (1) model ML YOLOv8-seg bila tersedia, lalu
    (2) fallback GrabCut + multi-strategi kontur.
    Mengembalikan (gambar_edges, kontur_dokumen|None, ratio, fg_mask|None, method).
    """
    ratio = image.shape[0] / 600.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 600))
    area_img = resized.shape[0] * resized.shape[1]
    rh, rw = resized.shape[:2]

    # ---- (1) Detektor ML lebih dulu ----
    quad_ml, conf = _try_ml_detector(image)
    if quad_ml is not None:
        doc_cnt = (np.asarray(quad_ml, np.float32) / ratio).reshape(4, 1, 2).astype(np.int32)
        fg = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(fg, [doc_cnt.reshape(4, 2)], 255)
        edged_vis = cv2.Canny(fg, 50, 150)
        return edged_vis, doc_cnt, ratio, fg, f'ml:{conf:.2f}'

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Hapus latar otomatis dulu agar deteksi tepi tak terganggu clutter ---
    fg = segment_foreground(resized)
    if fg is not None:
        # batasi area edge ke foreground; tepi mask jadi batas dokumen yang tegas
        gray = cv2.bitwise_and(gray, gray, mask=fg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    best_quad, best_score = None, -1.0
    edged_vis = None

    def consider(c):
        """Evaluasi satu kontur jadi quad, perbarui best bila lebih baik."""
        nonlocal best_quad, best_score
        if cv2.contourArea(c) < 0.10 * area_img:
            return
        peri = cv2.arcLength(c, True)
        for eps in (0.015, 0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                s = _quad_score(approx, area_img)
                if s > best_score:
                    best_score, best_quad = s, approx
                return
        # tak ada quad bersih -> convex hull lalu min-area-rect
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            s = _quad_score(approx, area_img)
            if s > best_score:
                best_score, best_quad = s, approx
        else:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            s = _quad_score(box.astype(np.int32), area_img)
            if s > best_score:
                best_score = s
                best_quad = box.reshape(4, 1, 2).astype(np.int32)

    # 1) Kandidat langsung dari kontur mask foreground (paling bersih)
    if fg is not None:
        mcnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(mcnts, key=cv2.contourArea, reverse=True)[:3]:
            consider(c)
        edged_vis = cv2.Canny(fg, 50, 150)

    # 2) Kandidat dari beberapa peta tepi (pada gambar yang sudah dibersihkan)
    for edged in _candidate_edges(gray):
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)
        if edged_vis is None:
            edged_vis = closed

        cnts, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            consider(c)

    doc_cnt = best_quad if best_score > 0.35 else None
    if edged_vis is None:
        edged_vis = np.zeros(gray.shape, np.uint8)

    return edged_vis, doc_cnt, ratio, fg, 'classic'


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


def to_grayscale_doc(image):
    """Grayscale dengan latar dinormalisasi agar bersih seperti scan abu-abu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # estimasi latar lewat blur besar, lalu bagi untuk meratakan pencahayaan
    bg = cv2.medianBlur(gray, 41)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def to_scanner_bw(image, strength=10):
    """
    Hitam-putih ala scanner: ratakan pencahayaan (background division) lalu
    adaptive threshold. Menghasilkan teks tajam di atas latar putih bersih.
    strength: makin tinggi makin agresif memutihkan latar (5-20).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ratakan pencahayaan: bagi dengan estimasi latar belakang
    bg = cv2.medianBlur(gray, 41)
    norm = cv2.divide(gray, bg, scale=255).astype(np.uint8)

    thr = cv2.adaptiveThreshold(norm, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, int(strength))
    # hilangkan bintik kecil
    thr = cv2.medianBlur(thr, 3)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
# Penyesuaian kontras & saturasi
# ----------------------------------------------------------------------------
def rotate_image(image, action):
    """
    Ubah orientasi gambar.
    action:
      'cw'    -> putar 90 derajat searah jarum jam
      'ccw'   -> putar 90 derajat berlawanan jarum jam
      '180'   -> putar 180 derajat
      'fliph' -> cermin horizontal (kiri-kanan)
      'flipv' -> cermin vertikal (atas-bawah)
    """
    if action == 'cw':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if action == 'ccw':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if action == '180':
        return cv2.rotate(image, cv2.ROTATE_180)
    if action == 'fliph':
        return cv2.flip(image, 1)
    if action == 'flipv':
        return cv2.flip(image, 0)
    return image


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

    edged, doc_cnt, ratio, fg, method = detect_document(img)

    h, w = img.shape[:2]

    # gambar overlay kontur tepi pada gambar asli (untuk visualisasi)
    overlay = img.copy()
    detected = doc_cnt is not None
    if detected:
        cv2.drawContours(overlay,
                         [(doc_cnt.reshape(4, 2) * ratio).astype(int)],
                         -1, (0, 220, 120), max(3, img.shape[1] // 250))

    # preview latar dihapus (foreground di atas putih)
    if fg is not None:
        fg_full = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
        nobg = img.copy()
        nobg[fg_full == 0] = (255, 255, 255)
    else:
        nobg = img.copy()

    scanned = make_scan(img, doc_cnt, ratio)
    scan_bw = scanner_effect(scanned)
    edges_vis = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    if doc_cnt is not None:
        corners = order_points(doc_cnt.reshape(4, 2) * ratio).tolist()
    else:
        corners = [[0, 0], [w, 0], [w, h], [0, h]]

    return jsonify({
        'detected': detected,
        'method': method,
        'bg_removed': fg is not None,
        'edges': cv2_to_b64(edges_vis),
        'overlay': cv2_to_b64(overlay),
        'nobg': cv2_to_b64(nobg),
        'scanned': cv2_to_b64(scanned),
        'scan_bw': cv2_to_b64(scan_bw),
        'corners': corners,
        'orig_size': [w, h],
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
    mode = data.get('mode', 'color')

    out = adjust_contrast_saturation(img, contrast, saturation, brightness)

    if mode == 'gray':
        out = to_grayscale_doc(out)
    elif mode == 'bw':
        threshold = int(data.get('threshold', 10))
        out = to_scanner_bw(out, strength=threshold)

    return jsonify({'result': cv2_to_b64(out)})


@app.route('/rotate', methods=['POST'])
def rotate():
    """Ubah orientasi gambar (putar/cermin) pada gambar ter-scan."""
    data = request.get_json()
    img = b64_to_cv2(data['image'])
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    action = data.get('action', '')
    out = rotate_image(img, action)
    return jsonify({'result': cv2_to_b64(out)})


@app.route('/warp', methods=['POST'])
def warp():
    """Koreksi perspektif manual dengan 4 titik sudut dari user."""
    data = request.get_json()
    img = b64_to_cv2(data['image'])
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    pts = np.array(data['corners'], dtype='float32')
    warped = four_point_transform(img, pts)
    scan_bw = scanner_effect(warped)
    return jsonify({
        'scanned': cv2_to_b64(warped),
        'scan_bw': cv2_to_b64(scan_bw),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
