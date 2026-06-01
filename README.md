# Image Enhancement (Web)

Minimal web app implementing the PRD: upload or select example images from the `LOLv2-real` dataset and run a deterministic classical image enhancement pipeline using OpenCV.

Quick start:

1. Create a Python virtualenv and activate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

2. Open http://127.0.0.1:5000/ in your browser.

Notes:
- The app expects the dataset folder `LOLv2-real` to live next to `app.py` (as in the repository).
- Uploaded files are kept temporarily in `static/uploads` and results in `static/results`. Files older than 1 hour are cleaned up automatically.
