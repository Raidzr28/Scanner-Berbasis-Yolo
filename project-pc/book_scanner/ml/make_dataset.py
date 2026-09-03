"""
Generator dataset deteksi dokumen (YOLOv8-seg) — komposit sintetis.

Strategi (sesuai skenario book-scanner):
  - LATAR  : foto asli dari web (Lorem Picsum, tanpa login) + fallback offline.
  - HALAMAN: dokumen/buku dirender prosedural (teks, judul, gutter dua-halaman).
  - KOMPOSIT: halaman di-warp perspektif acak + pencahayaan/bayangan/noise,
              ditempel ke latar. Ground-truth 4 sudut presisi sempurna.

Output (format YOLOv8 segmentasi):
  datasets/docseg/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt      # "0 x1 y1 x2 y2 x3 y3 x4 y4" (ternormalisasi)
  data.yaml

Jalankan:  python ml/make_dataset.py --n-train 300 --n-val 60
"""
import argparse
import io
import os
import random
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'datasets', 'docseg')

CANVAS_W, CANVAS_H = 640, 480


# ---------------------------------------------------------------------------
# Latar belakang: foto asli dari web, dengan fallback prosedural offline
# ---------------------------------------------------------------------------
def fetch_background(seed):
    """Ambil foto asli dari Lorem Picsum; fallback tekstur prosedural."""
    url = f"https://picsum.photos/seed/{seed}/{CANVAS_W}/{CANVAS_H}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = r.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (CANVAS_W, CANVAS_H))
    except Exception:
        pass
    return _procedural_background(seed)


def _procedural_background(seed):
    """Latar buatan (gradien + noise + bercak) bila unduhan gagal."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=3)
    grad = np.linspace(0, 1, CANVAS_H)[:, None, None]
    tilt = rng.integers(-40, 40, size=3)
    img = (base + grad * tilt).astype(np.float32)
    img = np.repeat(img, CANVAS_W, axis=1)
    img += rng.normal(0, 12, img.shape)
    # beberapa bercak / objek mengganggu
    for _ in range(rng.integers(3, 8)):
        c = (int(rng.integers(0, CANVAS_W)), int(rng.integers(0, CANVAS_H)))
        col = tuple(int(x) for x in rng.integers(0, 255, 3))
        cv2.circle(img, c, int(rng.integers(20, 90)), col, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Halaman dokumen / buku prosedural (RGBA, alpha penuh = area halaman)
# ---------------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_page(seed):
    rng = random.Random(seed)
    pw = rng.randint(360, 520)
    ph = rng.randint(460, 680)
    paper = rng.choice([(255, 255, 255), (250, 248, 240), (245, 245, 238),
                        (252, 250, 245)])
    img = Image.new('RGB', (pw, ph), paper)
    d = ImageDraw.Draw(img)

    margin = rng.randint(24, 48)
    two_page = rng.random() < 0.35  # tampilan buku terbuka dua halaman
    cols = [(margin, pw - margin)]
    if two_page:
        mid = pw // 2
        cols = [(margin, mid - 12), (mid + 12, pw - margin)]
        d.line([(mid, 0), (mid, ph)], fill=(200, 196, 188), width=2)

    ink = rng.choice([(30, 30, 30), (20, 20, 40), (40, 30, 30)])
    for (x0, x1) in cols:
        y = margin
        # judul
        if rng.random() < 0.8:
            th = rng.randint(22, 34)
            d.rectangle([x0, y, x0 + int((x1 - x0) * rng.uniform(.4, .9)), y + th],
                        fill=ink)
            y += th + rng.randint(14, 26)
        # baris teks (rectangle abu-abu mensimulasikan teks)
        while y < ph - margin:
            lh = rng.randint(7, 12)
            lw = int((x1 - x0) * rng.uniform(0.55, 1.0))
            shade = rng.randint(60, 130)
            d.rectangle([x0, y, x0 + lw, y + lh], fill=(shade, shade, shade))
            y += lh + rng.randint(6, 12)
            if rng.random() < 0.06:  # spasi paragraf
                y += rng.randint(10, 22)
        # kadang sisipkan gambar/figur
        if rng.random() < 0.4:
            fy = rng.randint(margin, max(margin, ph - 180))
            d.rectangle([x0, fy, x1, fy + rng.randint(80, 150)],
                        outline=(150, 150, 150), width=2,
                        fill=(rng.randint(200, 240),) * 3)

    # bingkai tepi halaman tipis (membantu model belajar batas)
    d.rectangle([0, 0, pw - 1, ph - 1], outline=(190, 188, 180), width=2)

    rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    alpha = np.full((ph, pw), 255, np.uint8)
    return rgb, alpha


# ---------------------------------------------------------------------------
# Komposit: warp perspektif acak + tempel ke latar + augmentasi
# ---------------------------------------------------------------------------
def random_quad(rng):
    """4 sudut tujuan di dalam kanvas dengan distorsi perspektif."""
    pad = 30
    # kotak dasar acak
    bw = rng.randint(int(CANVAS_W * 0.45), int(CANVAS_W * 0.85))
    bh = rng.randint(int(CANVAS_H * 0.45), int(CANVAS_H * 0.88))
    x0 = rng.randint(pad, max(pad, CANVAS_W - bw - pad))
    y0 = rng.randint(pad, max(pad, CANVAS_H - bh - pad))
    corners = np.float32([[x0, y0], [x0 + bw, y0],
                          [x0 + bw, y0 + bh], [x0, y0 + bh]])
    # jitter tiap sudut -> perspektif
    j = rng.uniform(0.0, 0.10) * min(bw, bh)
    corners += np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)]
                           for _ in range(4)])
    corners[:, 0] = np.clip(corners[:, 0], 2, CANVAS_W - 2)
    corners[:, 1] = np.clip(corners[:, 1], 2, CANVAS_H - 2)
    return corners


def composite(seed):
    rng = random.Random(seed * 7 + 1)
    nprng = np.random.default_rng(seed * 13 + 3)

    bg = fetch_background(seed).astype(np.float32)
    page, alpha = make_page(seed)
    ph, pw = alpha.shape

    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = random_quad(rng)
    M = cv2.getPerspectiveTransform(src, dst)

    warp = cv2.warpPerspective(page, M, (CANVAS_W, CANVAS_H)).astype(np.float32)
    wa = cv2.warpPerspective(alpha, M, (CANVAS_W, CANVAS_H)).astype(np.float32) / 255.0

    # bayangan halus di belakang halaman
    if rng.random() < 0.7:
        shadow = cv2.GaussianBlur(wa, (0, 0), rng.uniform(6, 14))
        off = (rng.randint(-8, 8), rng.randint(4, 14))
        shadow = np.roll(shadow, off, axis=(1, 0))[..., None]
        bg = bg * (1 - 0.35 * shadow)

    # pencahayaan tak rata pada halaman
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.15), CANVAS_W)
    gy = np.linspace(rng.uniform(0.8, 1.05), rng.uniform(0.85, 1.1), CANVAS_H)
    light = np.outer(gy, gx)[..., None]
    warp *= light

    wa3 = wa[..., None]
    out = warp * wa3 + bg * (1 - wa3)

    # augmentasi global
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.2))
    out += nprng.normal(0, rng.uniform(2, 9), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, dst


# ---------------------------------------------------------------------------
def write_split(split, n, start_seed):
    img_dir = os.path.join(OUT, 'images', split)
    lbl_dir = os.path.join(OUT, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n):
        seed = start_seed + i
        img, quad = composite(seed)
        name = f"{split}_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, name + '.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        # label YOLO-seg: class + polygon ternormalisasi
        norm = quad.copy()
        norm[:, 0] /= CANVAS_W
        norm[:, 1] /= CANVAS_H
        coords = ' '.join(f"{v:.6f}" for v in norm.reshape(-1))
        with open(os.path.join(lbl_dir, name + '.txt'), 'w') as f:
            f.write(f"0 {coords}\n")
        if (i + 1) % 25 == 0:
            print(f"  [{split}] {i + 1}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=300)
    ap.add_argument('--n-val', type=int, default=60)
    args = ap.parse_args()

    print(f"Membuat dataset di {OUT}")
    print(f"  train={args.n_train}  val={args.n_val}  ({CANVAS_W}x{CANVAS_H})")
    write_split('train', args.n_train, start_seed=1000)
    write_split('val', args.n_val, start_seed=900000)

    yaml_path = os.path.join(OUT, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(
            f"path: {OUT.replace(os.sep, '/')}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n  0: document\n"
        )
    print(f"Selesai. data.yaml -> {yaml_path}")


if __name__ == '__main__':
    main()
