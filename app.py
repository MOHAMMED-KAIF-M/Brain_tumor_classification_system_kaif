from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from functools import wraps
from openai import OpenAI
import os
from dotenv import load_dotenv
import requests

# ================= LOAD ENV =================
load_dotenv()

# ================= FLASK APP =================
app = Flask(__name__, template_folder='templates')
CORS(app)

# SESSION SECRET (NOT OpenAI KEY)
app.secret_key = "any_random_string_here_123"

# ================= OPENAI =================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# ================= GPU CONFIG =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled")
    except RuntimeError as e:
        print(e)

# ================= MODEL =================
MODEL_PATH = "model_best_final_1.h5"
IMAGE_SIZE = (224, 224)

print("🔹 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

class_map = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

# ================= LOGIN DECORATOR =================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ================= LOGIN =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'kaifu123' and request.form.get('password') == 'kaifu@123':
            session['username'] = 'kaifu123'
            return redirect(url_for('index'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ================= HOME =================
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = int(np.argmax(preds))

    pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        'label': class_map[idx],
        'confidence': float(np.max(preds)),
        'xai_image_url': f"data:image/png;base64,{img_base64}"
    })
# ================= CHATBOT =================
@app.post("/chat")
@login_required
def chat():
    data = request.json
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please enter a question."}), 400

    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
    if user_msg.lower() in greetings:
        return jsonify({"reply": "Hello! 😊 Ask me any biology or medical-related question."})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Biology-only assistant.\n"
                    "Answer ONLY biology, medical, anatomy, physiology, genetics, "
                    "neuroscience, brain tumors, healthcare questions.\n"
                    "If NOT biology related, reply exactly:\n"
                    "'I can only answer biology-related questions.'\n"
                    "Keep answers short and non-diagnostic."
                )
            },
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=150
    )

    return jsonify({"reply": response.choices[0].message.content})

@app.route('/brain-design')
def brain_design():
    return render_template('Brain_Design.html')

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)