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
    """Extract all text from image using EasyOCR."""
    results = ocr_reader.readtext(image_path, detail=0, paragraph=True)
    return results


# ─────────────────────────────────────────────
# Card Type Detection
# ─────────────────────────────────────────────
def detect_card_type(texts):
    full = " ".join(texts).upper()
    if "AADHAAR" in full or "UIDAI" in full or "UNIQUE IDENTIFICATION" in full:
        return "aadhar"
    elif "PERMANENT ACCOUNT" in full or ("INCOME TAX" in full and "PAN" in full):
        return "pan"
    elif "DRIVING" in full or "LICENCE" in full or "LICENSE" in full or "TRANSPORT" in full:
        return "driving"
    elif "VOTER" in full or "ELECTION" in full or "ELECTORS" in full or "EPIC" in full:
        return "voter"
    elif "PASSPORT" in full or "REPUBLIC OF INDIA" in full and "SURNAME" in full:
        return "passport"
    elif "EMPLOYEE" in full or "STAFF" in full or "OFFICE" in full or "ID CARD" in full or "COMPANY" in full:
        return "office"
    return "unknown"


# ─────────────────────────────────────────────
# Card Parsers
# ─────────────────────────────────────────────
def extract_dob(full_text):
    dob = re.search(r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b', full_text)
    return dob.group() if dob else None

def extract_name_near_keyword(texts, keywords):
    for i, line in enumerate(texts):
        for kw in keywords:
            if re.search(kw, line, re.IGNORECASE):
                # check same line after keyword
                cleaned = re.sub(kw, '', line, flags=re.IGNORECASE).strip(': ').strip()
                if len(cleaned) > 2 and not any(c.isdigit() for c in cleaned):
                    return cleaned
                # check next line
                if i + 1 < len(texts):
                    nxt = texts[i + 1].strip()
                    if len(nxt) > 2 and not any(c.isdigit() for c in nxt):
                        return nxt
    return None


def parse_aadhar(texts):
    full = " ".join(texts)
    data = {"card_type": "Aadhar Card", "name": None, "dob": None,
            "gender": None, "aadhar_number": None, "raw_text": texts}
    m = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', full)
    if m: data["aadhar_number"] = m.group()
    data["dob"] = extract_dob(full)
    if re.search(r'\bFEMALE\b', full, re.IGNORECASE): data["gender"] = "Female"
    elif re.search(r'\bMALE\b', full, re.IGNORECASE): data["gender"] = "Male"
    data["name"] = extract_name_near_keyword(texts, [r'DOB', r'Date of Birth', r'Year of Birth'])
    return data


def parse_pan(texts):
    full = " ".join(texts)
    data = {"card_type": "PAN Card", "name": None, "father_name": None,
            "dob": None, "pan_number": None, "raw_text": texts}
    m = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full)
    if m: data["pan_number"] = m.group()
    data["dob"] = extract_dob(full)
    data["name"] = extract_name_near_keyword(texts, [r"^Name", r"\bName\b"])
    data["father_name"] = extract_name_near_keyword(texts, [r"Father", r"Father'?s?\s*Name"])
    return data


def parse_driving(texts):
    full = " ".join(texts)
    data = {"card_type": "Driving Licence", "name": None, "dob": None,
            "dl_number": None, "address": None, "valid_till": None,
            "vehicle_class": None, "raw_text": texts}
    # DL number: e.g. MH0120230012345
    m = re.search(r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b', full)
    if m: data["dl_number"] = m.group()
    data["dob"] = extract_dob(full)
    data["name"] = extract_name_near_keyword(texts, [r'\bName\b', r'\bHolder\b'])
    # Valid till
    valid = re.search(r'(Valid|Validity|Exp)[^\d]*(\d{2}[\/\-]\d{2}[\/\-]\d{4})', full, re.IGNORECASE)
    if valid: data["valid_till"] = valid.group(2)
    # Vehicle class
    vc = re.search(r'(LMV|MCWG|HMV|MGV|Class)[^\n]*', full, re.IGNORECASE)
    if vc: data["vehicle_class"] = vc.group().strip()
    return data


def parse_voter(texts):
    full = " ".join(texts)
    data = {"card_type": "Voter ID Card", "name": None, "father_husband_name": None,
            "dob": None, "voter_id": None, "gender": None,
            "address": None, "raw_text": texts}
    # Voter ID: e.g. ABC1234567
    m = re.search(r'\b[A-Z]{3}[0-9]{7}\b', full)
    if m: data["voter_id"] = m.group()
    data["dob"] = extract_dob(full)
    if re.search(r'\bFEMALE\b', full, re.IGNORECASE): data["gender"] = "Female"
    elif re.search(r'\bMALE\b', full, re.IGNORECASE): data["gender"] = "Male"
    data["name"] = extract_name_near_keyword(texts, [r"\bName\b", r"\bElector'?s?\s*Name\b"])
    data["father_husband_name"] = extract_name_near_keyword(texts, [r"Father", r"Husband", r"Guardian"])
    return data


def parse_passport(texts):
    full = " ".join(texts)
    data = {"card_type": "Passport", "surname": None, "given_name": None,
            "dob": None, "passport_number": None, "nationality": None,
            "sex": None, "place_of_birth": None, "date_of_issue": None,
            "date_of_expiry": None, "raw_text": texts}
    # Passport number: e.g. A1234567
    m = re.search(r'\b[A-Z][0-9]{7}\b', full)
    if m: data["passport_number"] = m.group()
    data["dob"] = extract_dob(full)
    if re.search(r'\bF\b|\bFEMALE\b', full, re.IGNORECASE): data["sex"] = "Female"
    elif re.search(r'\bM\b|\bMALE\b', full, re.IGNORECASE): data["sex"] = "Male"
    data["surname"] = extract_name_near_keyword(texts, [r"Surname", r"Last\s*Name"])
    data["given_name"] = extract_name_near_keyword(texts, [r"Given\s*Name", r"First\s*Name"])
    data["nationality"] = "Indian" if "INDIAN" in full.upper() else None
    expiry = re.search(r'(Expiry|Expiration|Valid Until)[^\d]*(\d{2}[\/\-]\d{2}[\/\-]\d{4})', full, re.IGNORECASE)
    if expiry: data["date_of_expiry"] = expiry.group(2)
    return data


def parse_office(texts):
    full = " ".join(texts)
    data = {"card_type": "Office / Employee ID", "name": None, "employee_id": None,
            "designation": None, "department": None, "company": None,
            "email": None, "phone": None, "raw_text": texts}
    # Email
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full)
    if email: data["email"] = email.group()
    # Phone
    phone = re.search(r'\b[\+]?[\d\s\-]{10,13}\b', full)
    if phone: data["phone"] = phone.group().strip()
    # Employee ID
    emp = re.search(r'(EMP|ID|Staff|Employee\s*No)[^\w]*([A-Z0-9\-]+)', full, re.IGNORECASE)
    if emp: data["employee_id"] = emp.group(2)
    data["name"] = extract_name_near_keyword(texts, [r'\bName\b', r'\bEmployee\b'])
    data["designation"] = extract_name_near_keyword(texts, [r'Designation', r'Position', r'Role', r'Title'])
    data["department"] = extract_name_near_keyword(texts, [r'Department', r'Dept', r'Division'])
    # Company: usually first or second line
    for line in texts[:3]:
        if len(line) > 3 and not any(c.isdigit() for c in line):
            data["company"] = line.strip()
            break
    return data


# ─────────────────────────────────────────────
# Face Match (Lightweight - no dlib/deepface)
# ─────────────────────────────────────────────
def match_faces(card_image_path, selfie_image_path):
    try:
        card_img = Image.open(card_image_path).convert("RGB").resize((200, 200))
        selfie_img = Image.open(selfie_image_path).convert("RGB").resize((200, 200))

        card_arr = np.array(card_img).astype(float) / 255.0
        selfie_arr = np.array(selfie_img).astype(float) / 255.0

        correlation = np.corrcoef(card_arr.flatten(), selfie_arr.flatten())[0, 1]
        similarity = round(float(correlation) * 100, 2)
        similarity = max(0, min(100, similarity))
        match = similarity > 60

        return {
            "match": match,
            "similarity": similarity,
            "status": "Face Matched!" if match else "Face Not Matched"
        }
    except Exception as e:
        return {"match": False, "similarity": 0, "status": f"Comparison failed: {str(e)}"}


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
        "filename": filename, "label": label,
        "confidence": confidence, "top3": top3,
        "image_url": f"/static/uploads/{filename}"
    })


@app.route("/ocr", methods=["POST"])
def ocr_extract():
    try:
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

        # Run OCR
        texts = extract_ocr_text(card_path)

        # Detect and parse
        card_type = detect_card_type(texts)
        parsers = {
            "aadhar":   parse_aadhar,
            "pan":      parse_pan,
            "driving":  parse_driving,
            "voter":    parse_voter,
            "passport": parse_passport,
            "office":   parse_office,
        }
        parsed = parsers.get(card_type, lambda t: {"card_type": "Unknown Card", "raw_text": t})(texts)

        # Face match
        face_result = None
        if selfie_path:
            face_result = match_faces(card_path, selfie_path)

        return jsonify({
            "card_data": parsed,
            "card_type": card_type,
            "card_image_url": f"/static/uploads/{card_filename}",
            "selfie_image_url": f"/static/uploads/{selfie_filename}" if selfie_path else None,
            "face_match": face_result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history():
    rows = get_all_results()
    return jsonify([
        {"id": r[0], "filename": r[1], "label": r[2],
         "confidence": round(r[3], 2), "timestamp": r[4]}
        for r in rows
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
