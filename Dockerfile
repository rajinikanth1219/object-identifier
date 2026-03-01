FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p static/uploads
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:8080 --timeout 120 --workers 1 app:app
```

---

**Step 2 — Add gunicorn to `requirements.txt`:**
```
flask==2.3.3
torch==2.0.1+cpu
torchvision==0.15.2+cpu
pillow==10.0.0
werkzeug==2.3.7
numpy==1.24.3
easyocr==1.7.1
opencv-python-headless==4.8.0.76
gunicorn==21.2.0
--extra-index-url https://download.pytorch.org/whl/cpu