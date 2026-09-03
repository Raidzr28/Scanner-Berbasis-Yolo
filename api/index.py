"""Vercel serverless entrypoint.

Serves the Flask app that lives in ``project PC/book_scanner/``. torch /
ultralytics are intentionally absent from requirements.txt, so ``ml.infer``'s
YOLO path degrades to the GrabCut + contour fallback on its own — the /scan
response shape is identical either way.

Vercel's @vercel/python builder auto-detects the module-level ``app`` (a WSGI
callable) and wraps it.
"""
import os
import sys

_APP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "project-pc", "book_scanner")
)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from app import app  # noqa: E402,F401
