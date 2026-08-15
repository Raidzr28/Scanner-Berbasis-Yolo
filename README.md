# ScanBook — Scanner Dokumen Berbasis YOLOv8-seg

Aplikasi web **Flask + OpenCV** untuk mengubah foto buku/dokumen menjadi hasil
scan yang rapi: deteksi tepi otomatis, koreksi perspektif, penghapusan latar,
serta pengaturan kontras, saturasi, dan kecerahan secara real-time.

Deteksi sudut dokumen memakai model **YOLOv8n-seg** yang di-*fine-tune* pada
dataset komposit sintetis, dengan *fallback* otomatis ke metode klasik
(GrabCut + multi-strategi kontur) bila model atau `ultralytics` tidak tersedia.

---

## Fitur

| Fitur | Keterangan |
|---|---|
| Deteksi tepi | YOLOv8n-seg → mask → 4 sudut. Fallback: GrabCut + Canny/morphological gradient + skoring quad |
| Efek scanner | `getPerspectiveTransform` + `warpPerspective` (four-point transform) |
| Hapus latar | Segmentasi foreground GrabCut, latar dijadikan putih bersih |
| Koreksi manual | Geser 4 titik sudut sendiri bila deteksi otomatis meleset |
| Rotasi & cermin | 90° CW/CCW, 180°, flip horizontal/vertikal |
| Mode output | Berwarna, Grayscale (background division), Hitam-Putih (adaptive threshold) |
| Penyesuaian | Slider kontras (0.5–3.0), saturasi (0.0–3.0), kecerahan (−100–100) |

---

## Instalasi

```bash
cd "project PC/book_scanner"
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Dependensi inti: `flask`, `opencv-python-headless`, `numpy`, `pillow`.

### Opsional — mengaktifkan detektor YOLO

Detektor ML memerlukan paket tambahan yang **tidak** ada di `requirements.txt`
(agar instalasi dasar tetap ringan). Tanpa paket ini aplikasi tetap berjalan
normal memakai jalur klasik:

```bash
pip install ultralytics torch torchvision
```

---

## Menjalankan

```bash
python app.py
```

Buka **http://127.0.0.1:5000**

> Secara default server mendengarkan di `0.0.0.0:5000`, sehingga dapat diakses
> dari perangkat lain di jaringan yang sama. Untuk penggunaan lokal saja, ubah
> ke `host='127.0.0.1'` di bagian bawah `app.py`.

---

## Alur Pemakaian

1. Unggah gambar buku (seret & lepas atau klik).
2. Klik **Deteksi Tepi & Scan** — hasil muncul dalam beberapa tab:
   *Tepi Terdeteksi*, *Peta Tepi*, *Tanpa Latar*, *Hasil Scan*, *Mode B/W*.
3. Bila sudut meleset, gunakan koreksi manual 4 titik lalu **Terapkan**.
4. Atur rotasi, mode warna, dan slider penyesuaian.
5. **Unduh Hasil**.

---

## Struktur Proyek

```
Scanner Berbasis Yolo/
├── .gitignore
├── README.md
└── project PC/
    ├── app.py                    # versi awal (deteksi klasik saja, arsip)
    ├── index.html                # UI versi awal
    └── book_scanner/             # === versi aktif ===
        ├── app.py                # backend Flask + pipeline OpenCV
        ├── requirements.txt
        ├── templates/
        │   └── index.html        # UI
        └── ml/
            ├── infer.py          # inferensi YOLOv8-seg (lazy load, thread-safe)
            ├── train.py          # fine-tune YOLOv8n-seg
            ├── make_dataset.py   # generator dataset komposit sintetis
            └── weights/
                └── book_seg.pt   # bobot hasil training
```

Direktori `datasets/` dan `ml/runs/` sengaja **tidak** disertakan di repositori
(lihat `.gitignore`) karena dapat dibuat ulang dan berisi path absolut mesin
lokal. Lihat bagian *Melatih Ulang Model* untuk membuatnya kembali.

---

## API Endpoint

Semua endpoint menerima dan mengembalikan JSON. Gambar dikirim sebagai
data URL base64 (`data:image/png;base64,...`). Batas ukuran request 16 MB.

| Method | Route | Body | Response |
|---|---|---|---|
| `GET` | `/` | — | Halaman UI |
| `POST` | `/scan` | `image` | `detected`, `method`, `bg_removed`, `edges`, `overlay`, `nobg`, `scanned`, `scan_bw`, `corners`, `orig_size` |
| `POST` | `/adjust` | `image`, `contrast`, `saturation`, `brightness`, `mode` (`color`\|`gray`\|`bw`), `threshold` | `result` |
| `POST` | `/rotate` | `image`, `action` (`cw`\|`ccw`\|`180`\|`fliph`\|`flipv`) | `result` |
| `POST` | `/warp` | `image`, `corners` (4×2) | `scanned`, `scan_bw` |

Field `method` pada `/scan` bernilai `ml:0.87` bila deteksi berasal dari model
YOLO (angka = confidence), atau `classic` bila memakai jalur fallback.

---

## Melatih Ulang Model

```bash
cd "project PC/book_scanner"

# 1. Bangun dataset komposit sintetis
#    Latar diambil dari Lorem Picsum (butuh internet), halaman dirender
#    prosedural, lalu di-warp perspektif acak. Ground-truth 4 sudut presisi.
python ml/make_dataset.py --n-train 300 --n-val 60

# 2. Fine-tune YOLOv8n-seg (CPU-friendly)
python ml/train.py --epochs 25 --imgsz 512 --batch 8
```

Bobot pra-terlatih `yolov8n-seg.pt` diunduh otomatis oleh `ultralytics`.
Model terbaik disalin ke `ml/weights/book_seg.pt` dan langsung dipakai
`ml/infer.py` pada permintaan berikutnya.

---

## Catatan Teknis

- **Skoring quad** menggabungkan luas relatif, keseimbangan sisi, dan
  penalti penyimpangan sudut dari 90° — dipakai memilih kandidat kontur terbaik.
- **Grayscale** memakai *background division* (`medianBlur` 41 px lalu
  `cv2.divide`) untuk meratakan pencahayaan sebelum normalisasi.
- **Hitam-putih** memakai *background division* + `adaptiveThreshold` Gaussian,
  parameter `strength` (5–20) mengatur agresivitas pemutihan latar.
- **Saturasi** diatur pada kanal S ruang warna HSV; kontras dan kecerahan
  diterapkan langsung pada piksel BGR.
- Model dimuat sekali secara *lazy* dengan `threading.Lock`, dan gagal secara
  senyap ke jalur klasik bila bobot atau `torch` tidak ada.

---

## Keamanan

Aplikasi ini ditujukan untuk **penggunaan lokal / demo**. Sebelum
menjalankannya di lingkungan publik, perhatikan:

- Server bawaan Flask bukan untuk produksi — gunakan `gunicorn` atau `waitress`.
- Endpoint tidak memiliki autentikasi, rate limiting, maupun validasi skema
  input. Nilai `corners` pada `/warp` dipakai langsung untuk menentukan ukuran
  kanvas hasil, sehingga koordinat ekstrem dapat memicu alokasi memori besar.
- Gambar yang di-decode tidak dibatasi resolusinya (hanya ukuran request yang
  dibatasi 16 MB), sehingga rentan terhadap *decompression bomb*.
- Berkas `.pt` dimuat lewat `torch.load` (pickle). Jangan pernah memuat bobot
  dari sumber yang tidak dipercaya.

---

## Lisensi

Belum ditentukan. Tambahkan berkas `LICENSE` sebelum publikasi bila diperlukan.

Catatan: gambar latar dataset diambil dari [Lorem Picsum](https://picsum.photos)
(bersumber dari Unsplash). Periksa ketentuan lisensinya bila dataset hasil
generate hendak didistribusikan ulang.
