import os
import json
import re
import urllib.request
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import easyocr
from deepface import DeepFace
from database import init_db, save_result, get_all_results

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─────────────────────────────────────────────
# Load ImageNet Labels
# ─────────────────────────────────────────────
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_FILE = "imagenet_labels.json"
if not os.path.exists(LABELS_FILE):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)
with open(LABELS_FILE) as f:
    LABELS = json.load(f)

# ─────────────────────────────────────────────
# Load CNN Model (MobileNetV2)
# ─────────────────────────────────────────────
print("Loading CNN model...")
cnn_model = models.mobilenet_v2(pretrained=True)
cnn_model.eval()
print("CNN model loaded!")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# Load EasyOCR
# ─────────────────────────────────────────────
print("Loading OCR engine...")
ocr_reader = easyocr.Reader(['en'], gpu=False)
print("OCR engine loaded!")

# Initialize DB
init_db()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_prob, top3_idx = torch.topk(probs, 3)
    top3 = [{"label": LABELS[idx.item()].replace("_", " ").title(),
              "confidence": round(prob.item() * 100, 2)}
            for prob, idx in zip(top3_prob, top3_idx)]
    return top3[0]["label"], top3[0]["confidence"], top3


def extract_ocr_text(image_path):
    results = ocr_reader.readtext(image_path, detail=0, paragraph=True)
    return results


def parse_aadhar(texts):
    full_text = " ".join(texts)
    data = {
        "card_type": "Aadhar Card",
        "name": None,
        "dob": None,
        "gender": None,
        "aadhar_number": None,
        "raw_text": texts
    }
    aadhar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', full_text)
    if aadhar_match:
        data["aadhar_number"] = aadhar_match.group()
    dob_match = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', full_text)
    if dob_match:
        data["dob"] = dob_match.group()
    if re.search(r'\bMALE\b', full_text, re.IGNORECASE):
        data["gender"] = "Male"
    elif re.search(r'\bFEMALE\b', full_text, re.IGNORECASE):
        data["gender"] = "Female"
    for i, line in enumerate(texts):
        if re.search(r'(DOB|Date of Birth|Birth)', line, re.IGNORECASE) and i > 0:
            candidate = texts[i - 1].strip()
            if len(candidate) > 3 and not any(c.isdigit() for c in candidate):
                data["name"] = candidate
                break
        if re.search(r'^Name[:\s]', line, re.IGNORECASE):
            data["name"] = re.sub(r'^Name[:\s]*', '', line, flags=re.IGNORECASE).strip()
    return data


def parse_pan(texts):
    full_text = " ".join(texts)
    data = {
        "card_type": "PAN Card",
        "name": None,
        "father_name": None,
        "dob": None,
        "pan_number": None,
        "raw_text": texts
    }
    pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full_text)
    if pan_match:
        data["pan_number"] = pan_match.group()
    dob_match = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', full_text)
    if dob_match:
        data["dob"] = dob_match.group()
    for i, line in enumerate(texts):
        if re.search(r"father'?s?\s*name", line, re.IGNORECASE):
            if i + 1 < len(texts):
                data["father_name"] = texts[i + 1].strip()
        if re.search(r'\bname\b', line, re.IGNORECASE) and not re.search(r'father', line, re.IGNORECASE):
            if i + 1 < len(texts):
                candidate = texts[i + 1].strip()
                if len(candidate) > 2:
                    data["name"] = candidate
    return data


def detect_card_type(texts):
    full_text = " ".join(texts).upper()
    if "AADHAAR" in full_text or "UIDAI" in full_text or "UNIQUE IDENTIFICATION" in full_text:
        return "aadhar"
    elif "INCOME TAX" in full_text or "PERMANENT ACCOUNT" in full_text:
        return "pan"
    elif "PAN" in full_text:
        return "pan"
    return "unknown"


def match_faces(card_image_path, selfie_image_path):
    try:
        result = DeepFace.verify(
            img1_path=card_image_path,
            img2_path=selfie_image_path,
            model_name="VGG-Face",
            enforce_detection=False
        )
        match = result.get("verified", False)
        distance = result.get("distance", 1.0)
        similarity = round((1 - distance) * 100, 2)
        similarity = max(0, min(100, similarity))
        return {
            "match": match,
            "similarity": similarity,
            "status": "Face Matched!" if match else "Face Not Matched"
        }
    except Exception as e:
        return {"match": False, "similarity": 0, "status": f"Face detection failed: {str(e)}"}


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
        return jsonify({"error": "File type not allowed."}), 400
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


@app.route("/ocr", methods=["POST"])
def ocr_extract():
    if "card" not in request.files:
        return jsonify({"error": "No card image provided"}), 400

    card_file = request.files["card"]
    selfie_file = request.files.get("selfie")

    card_filename = "card_" + secure_filename(card_file.filename)
    card_path = os.path.join(app.config["UPLOAD_FOLDER"], card_filename)
    card_file.save(card_path)

    selfie_path = None
    selfie_filename = None
    if selfie_file and selfie_file.filename:
        selfie_filename = "selfie_" + secure_filename(selfie_file.filename)
        selfie_path = os.path.join(app.config["UPLOAD_FOLDER"], selfie_filename)
        selfie_file.save(selfie_path)

    try:
        texts = extract_ocr_text(card_path)
    except Exception as e:
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    card_type = detect_card_type(texts)
    if card_type == "aadhar":
        parsed = parse_aadhar(texts)
    elif card_type == "pan":
        parsed = parse_pan(texts)
    else:
        parsed = {"card_type": "Unknown Card", "raw_text": texts}

    face_result = None
    if selfie_path:
        face_result = match_faces(card_path, selfie_path)

    return jsonify({
        "card_data": parsed,
        "card_image_url": f"/static/uploads/{card_filename}",
        "selfie_image_url": f"/static/uploads/{selfie_filename}" if selfie_path else None,
        "face_match": face_result
    })


@app.route("/history")
def history():
    rows = get_all_results()
    return jsonify([
        {"id": r[0], "filename": r[1], "label": r[2],
         "confidence": round(r[3], 2), "timestamp": r[4]}
        for r in rows
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
