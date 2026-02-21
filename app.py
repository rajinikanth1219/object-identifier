import os
import json
import urllib.request
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from database import init_db, save_result, get_all_results

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─────────────────────────────────────────────
# Download ImageNet class labels
# ─────────────────────────────────────────────
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_FILE = "imagenet_labels.json"

if not os.path.exists(LABELS_FILE):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)

with open(LABELS_FILE) as f:
    LABELS = json.load(f)

# ─────────────────────────────────────────────
# Load MobileNetV2 via PyTorch (lightweight, CPU friendly)
# ─────────────────────────────────────────────
print("Loading CNN model (MobileNetV2 via PyTorch)...")
model = models.mobilenet_v2(pretrained=True)
model.eval()
print("Model loaded successfully!")

# ─────────────────────────────────────────────
# Image Preprocessing Pipeline
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Initialize DB
init_db()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """
    Run CNN prediction on the image.
    Returns top label, confidence %, and top-3 predictions.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_prob, top3_idx = torch.topk(probabilities, 3)

    top3 = []
    for prob, idx in zip(top3_prob, top3_idx):
        label = LABELS[idx.item()].replace("_", " ").title()
        confidence = round(prob.item() * 100, 2)
        top3.append({"label": label, "confidence": confidence})

    return top3[0]["label"], top3[0]["confidence"], top3


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    history = get_all_results()
    history_list = [
        {"id": row[0], "filename": row[1], "label": row[2],
         "confidence": round(row[3], 2), "timestamp": row[4]}
        for row in history
    ]
    return render_template("index.html", history=history_list)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use PNG, JPG, JPEG, GIF, BMP, or WEBP."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        label, confidence, top3 = predict_image(save_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    save_result(filename, label, confidence)

    return jsonify({
        "filename": filename,
        "label": label,
        "confidence": confidence,
        "top3": top3,
        "image_url": f"/static/uploads/{filename}"
    })


@app.route("/history")
def history():
    rows = get_all_results()
    return jsonify([
        {"id": r[0], "filename": r[1], "label": r[2],
         "confidence": round(r[3], 2), "timestamp": r[4]}
        for r in rows
    ])


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
