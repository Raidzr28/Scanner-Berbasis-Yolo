"""
Fine-tune YOLOv8n-seg untuk deteksi dokumen (1 kelas: document).

Memakai bobot pra-terlatih COCO (yolov8n-seg.pt) lalu dilatih ulang pada
dataset komposit di datasets/docseg. CPU-only friendly.

Jalankan:  python ml/train.py --epochs 25 --imgsz 512
Hasil terbaik akan disalin ke ml/weights/book_seg.pt
"""
import argparse
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'datasets', 'docseg', 'data.yaml')
WEIGHTS_DIR = os.path.join(HERE, 'weights')
FINAL = os.path.join(WEIGHTS_DIR, 'book_seg.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--model', default='yolov8n-seg.pt')
    args = ap.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    model = YOLO(args.model)  # unduh bobot pra-terlatih bila belum ada
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device='cpu',
        project=os.path.join(HERE, 'runs'),
        name='book_seg',
        exist_ok=True,
        patience=10,
        # augmentasi ringan; latar sudah bervariasi dari komposit
        degrees=5.0, translate=0.06, scale=0.3, fliplr=0.5,
        mosaic=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        verbose=True,
    )

    best = os.path.join(HERE, 'runs', 'book_seg', 'weights', 'best.pt')
    if os.path.exists(best):
        shutil.copy(best, FINAL)
        print(f"\nModel terbaik disalin -> {FINAL}")
    else:
        print("PERINGATAN: best.pt tidak ditemukan.")


if __name__ == '__main__':
    main()
