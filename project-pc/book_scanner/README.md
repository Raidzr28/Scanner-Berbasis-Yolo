# ScanBook — book_scanner (versi aktif)

Backend Flask + OpenCV dengan deteksi sudut dokumen YOLOv8n-seg dan fallback
klasik (GrabCut + multi-strategi kontur).

Dokumentasi lengkap — instalasi, endpoint API, cara melatih ulang model, dan
catatan keamanan — ada di [README utama](../../README.md) di root repositori.

## Ringkas

```bash
pip install -r requirements.txt        # inti
pip install ultralytics torch torchvision   # opsional, mengaktifkan detektor YOLO
python app.py                          # http://127.0.0.1:5000
```

## Isi

```
book_scanner/
├── app.py                # Flask: /, /scan, /adjust, /rotate, /warp
├── requirements.txt
├── templates/index.html  # UI
└── ml/
    ├── infer.py          # inferensi YOLOv8-seg (lazy load, thread-safe)
    ├── train.py          # fine-tune YOLOv8n-seg
    ├── make_dataset.py   # generator dataset komposit sintetis
    └── weights/book_seg.pt
```

`datasets/` dan `ml/runs/` tidak di-commit — buat ulang dengan
`python ml/make_dataset.py` lalu `python ml/train.py`.
