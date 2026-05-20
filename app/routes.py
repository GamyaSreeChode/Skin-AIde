from flask import request, render_template
from werkzeug.utils import secure_filename
from app import app
import os
import re
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from pathlib import Path

# Load .env file manually to handle encoding issues
env_path = Path('.env')
if env_path.exists():
    try:
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIRS = [BASE_DIR, BASE_DIR.parent]

def resolve_model_path(filename):
    for model_dir in MODEL_DIRS:
        candidate = model_dir / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Model file not found: {filename}")

# 🔥 Load model ONCE (very important)
model = tf.keras.models.load_model(
    resolve_model_path("HAM10000_Ensemble_91plus.keras"),
    compile=False
)

specialist_model = tf.keras.models.load_model(
    resolve_model_path("specialist_bcc_bkl.keras"),
    compile=False
)

WEAK_IDX = [1, 2] # Basal cell carcinoma, Benign keratosis
CONFIDENCE_THRESHOLD = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "60"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class_names = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevus',
    'Vascular lesion'
]

def get_medical_details(disease):
    cancer_types = ["Melanoma","Basal cell carcinoma","Actinic keratoses"]

    if disease in cancer_types:
        return "Yes", "High Risk", "Consult a dermatologist immediately."
    else:
        return "No", "Low Risk", "Maintain skincare and monitor changes."

def clean_precautions_text(text):
    """Normalize LLM output into plain readable text for UI display."""
    cleaned_text = text.replace("\r\n", "\n")
    cleaned_text = re.sub(r"^#{1,6}\s*", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = cleaned_text.replace("**", "")
    cleaned_text = re.sub(r"^\s*[-*•]+\s*", "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    section_names = [
        "Diet recommendations",
        "Foods to avoid",
        "Skincare routine",
        "Lifestyle changes",
        "When to see a specialist",
    ]

    formatted_sections = []
    for section_name in section_names:
        pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?=(?:{'|'.join(re.escape(name) for name in section_names)})\s*:|\Z)"
        match = re.search(pattern, cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        body = re.sub(r"\s+", " ", match.group(1)).strip()
        compact_points = [part.strip(" -") for part in body.split("-") if part.strip()]
        if len(compact_points) < 2:
            compact_points = [sentence.strip().rstrip(".!?") for sentence in re.split(r"(?<=[.!?])\s+", body) if sentence.strip()]

        compact_points = compact_points[:3]
        if not compact_points:
            continue

        bullet_lines = "\n".join(f"- {point}" for point in compact_points)
        formatted_sections.append(f"{section_name}:\n{bullet_lines}")

    if formatted_sections:
        return "\n\n".join(formatted_sections)

    return cleaned_text

def get_precautions_dynamic(disease, severity):
    """Generate dynamic precautions using Groq API based on disease and severity level"""
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    
    if not groq_api_key:
        app.logger.error("❌ GROQ_API_KEY not set in environment")
        return "❌ API Configuration Error: GROQ_API_KEY not found. Precautions unavailable."
    
    try:
        groq_url = "https://api.groq.com/openai/v1/chat/completions"

        severity_context = f"Severity level: {severity}." if severity != "Uncertain" else "Severity level: uncertain."

        prompt = (
            f"You are generating patient-friendly skin-care guidance.\n"
            f"Condition: {disease}.\n"
            f"{severity_context}\n"
            f"Keep it factual, clear, and specific to the condition and severity.\n"
            f"Return plain text only. No markdown, no bold, no ### headings.\n"
            f"Use exactly this compact format and nothing else:\n\n"
            f"Diet recommendations:\n"
            f"- point one\n"
            f"- point two\n"
            f"- point three\n\n"
            f"Foods to avoid:\n"
            f"- point one\n"
            f"- point two\n"
            f"- point three\n\n"
            f"Skincare routine:\n"
            f"- point one\n"
            f"- point two\n"
            f"- point three\n\n"
            f"Lifestyle changes:\n"
            f"- point one\n"
            f"- point two\n"
            f"- point three\n\n"
            f"When to see a specialist:\n"
            f"- point one\n"
            f"- point two\n"
            f"- point three\n\n"
            f"Give 2 to 3 bullets per section. Each bullet should be short and useful."
        )

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 220
        }

        response = requests.post(groq_url, headers=headers, json=data, timeout=15)
        app.logger.info(f"📡 Groq API Response Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                precautions = clean_precautions_text(result["choices"][0]["message"]["content"])
                app.logger.info(f"✅ Precautions from Groq for {disease} ({severity})")
                return precautions
            else:
                app.logger.error(f"❌ Invalid Groq response structure: {result}")
                return f"❌ API Error: Invalid response from Groq API"
        else:
            error_msg = response.text[:300]
            app.logger.error(f"❌ Groq API Error {response.status_code}: {error_msg}")
            return f"❌ API Failed: {response.status_code} - Model {GROQ_MODEL} could not generate precautions"
            
    except requests.exceptions.Timeout:
        app.logger.error(f"❌ Groq API timeout (15s exceeded)")
        return "❌ API Error: Request timeout - Groq API took too long to respond"
    except requests.exceptions.RequestException as e:
        app.logger.error(f"❌ Groq API connection error: {str(e)[:200]}")
        return f"❌ API Error: Connection failed - {str(e)[:100]}"
    except Exception as e:
        app.logger.error(f"❌ Groq exception: {str(e)[:200]}")
        return f"❌ API Error: {str(e)[:100]}"

@app.route('/', methods=['GET','POST'])
def home_page():
    res = None

    if request.method == 'POST':

        f = request.files['file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(path)

        # 🧠 Predict directly (no subprocess)
        img = Image.open(path).convert("RGB").resize((300,300))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        class_index = np.argmax(preds)

        # Routing to specialist model if prediction is BCC or BKL
        if class_index in WEAK_IDX:
            spec_preds = specialist_model.predict(img)[0]
            spec_class_index = np.argmax(spec_preds)
            final_class_index = WEAK_IDX[spec_class_index]
            confidence = round(float(spec_preds[spec_class_index] * 100), 2)
            disease = class_names[final_class_index]
        else:
            confidence = round(float(preds[class_index] * 100), 2)
            disease = class_names[class_index]

        '''if confidence < CONFIDENCE_THRESHOLD:
            disease = "No clear skin disease detected"
            cancer = "Unknown"
            severity = "Uncertain"
            advice = (
                "Prediction confidence is low. This may be normal skin or an unsupported image. "
                "Please upload a clearer close-up image and consult a dermatologist if concerned."
            )
            precautions = (
                "Use clear lighting, keep the lesion centered, and avoid blurry or distant photos. "
                "For medical concerns, seek professional diagnosis."
            )
        else:
            cancer, severity, advice = get_medical_details(disease)

            # Generate dynamic precautions based on disease AND severity level
            precautions = get_precautions_dynamic(disease, severity)'''
        # Always show predicted disease
        cancer, severity, advice = get_medical_details(disease)

        # If confidence is low, slightly adjust the advice
        if confidence < CONFIDENCE_THRESHOLD:
            advice = advice + " (Prediction confidence is low. Please verify with a dermatologist.)"

        # Generate precautions
        precautions = get_precautions_dynamic(disease, severity)

        res = {
            "disease": disease,
            "confidence": confidence,
            "cancer": cancer,
            "severity": severity,
            "advice": advice,
            "precautions": precautions
        }

    return render_template("index.html", res=res)