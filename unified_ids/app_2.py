"""
Unified Intrusion Detection System
Supports: NSL-KDD  |  CIC-IDS2017
Auth:     Face recognition + Password (both must pass)

Run: python app.py
"""

import os, sys, pickle, json, base64, secrets
import numpy as np
import pandas as pd
import tensorflow as tf
import face_recognition
import bcrypt
import joblib
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)   # random each restart; fine for single-admin use

# ── Paths ──────────────────────────────────────────────────────
BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDD_MODELS_DIR = os.path.join(BASE, 'ids', 'models', 'saved_models')
CIC_MODELS_DIR = os.path.join(BASE, 'IDS with DL - CIC2017', 'models')
USERS_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')

# ── Inline DataPreprocessor so joblib can unpickle KDD prep ───
class DataPreprocessor(BaseEstimator):
    def __init__(self):
        self.label_encoders   = {}
        self.scaler           = StandardScaler()
        self.feature_columns  = None
        self.col_names        = []
        self.categorical_cols = ['protocol_type', 'service', 'flag']

# ─────────────────────────────────────────────────────────────
#  Load ML models at startup
# ─────────────────────────────────────────────────────────────
MODELS = {}

try:
    print("Loading NSL-KDD ...")
    sys.path.insert(0, os.path.join(BASE, 'ids'))
    kdd_model = tf.keras.models.load_model(
        os.path.join(KDD_MODELS_DIR, 'best_model.keras'), compile=False)
    prep = joblib.load(os.path.join(KDD_MODELS_DIR, 'preprocessor.pkl'))
    sys.path.pop(0)
    MODELS['kdd'] = {
        'model': kdd_model, 'scaler': prep.scaler,
        'feature_names': prep.feature_columns, 'encoders': prep.label_encoders,
    }
    print(f"  ✓ NSL-KDD ready  ({len(prep.feature_columns)} features)")
except Exception as e:
    print(f"  ✗ NSL-KDD failed: {e}")

try:
    print("Loading CIC-IDS2017 ...")
    cic_model = tf.keras.models.load_model(
        os.path.join(CIC_MODELS_DIR, 'best_model.h5'))
    with open(os.path.join(CIC_MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        cic_scaler = pickle.load(f)
    with open(os.path.join(CIC_MODELS_DIR, 'feature_columns.pkl'), 'rb') as f:
        cic_cols = pickle.load(f)
    cic_names    = list(cic_cols.keys()) if isinstance(cic_cols, dict) else list(cic_cols)
    cic_encoders = cic_cols if isinstance(cic_cols, dict) else {}
    MODELS['cic'] = {
        'model': cic_model, 'scaler': cic_scaler,
        'feature_names': cic_names, 'encoders': cic_encoders,
    }
    print(f"  ✓ CIC-IDS2017 ready  ({len(cic_names)} features)")
except Exception as e:
    print(f"  ✗ CIC-IDS2017 failed: {e}")

if not MODELS:
    raise RuntimeError("No ML models loaded — check folder paths.")

MODEL_LABELS = {'kdd': 'NSL-KDD', 'cic': 'CIC-IDS2017'}

# ─────────────────────────────────────────────────────────────
#  Auth helpers
# ─────────────────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)


def verify_password(username, password):
    users = load_users()
    if username not in users:
        return False
    stored = users[username]['password_hash'].encode()
    return bcrypt.checkpw(password.encode(), stored)


def verify_face(username, image_b64):
    """Compare webcam snapshot (base64 JPEG) to stored face encoding."""
    users = load_users()
    if username not in users:
        return False

    stored_enc = np.array(users[username]['face_encoding'])

    # Decode base64 image
    try:
        header, data = image_b64.split(',', 1)
        img_bytes = base64.b64decode(data)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        import cv2
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return False

    locations = face_recognition.face_locations(rgb)
    if not locations:
        return False

    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        return False

    # Compare — tolerance 0.5 is default; lower = stricter
    results = face_recognition.compare_faces([stored_enc], encodings[0], tolerance=0.5)
    return bool(results[0])


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────
#  ML helpers
# ─────────────────────────────────────────────────────────────
def encode_categorical(encoders, col, value):
    le = encoders.get(col)
    if le is None:
        return 0
    value_str = str(value).strip()
    for cls in le.classes_:
        if value_str.lower() == cls.lower():
            return int(le.transform([cls])[0])
    return int(le.transform([le.classes_[0]])[0])


def preprocess(data, model_key):
    cfg           = MODELS[model_key]
    feature_names = cfg['feature_names']
    encoders      = cfg['encoders']
    scaler        = cfg['scaler']
    row = {}
    for col in feature_names:
        val = data.get(col, 0)
        if col in encoders:
            row[col] = encode_categorical(encoders, col, val)
        else:
            try:    row[col] = float(val)
            except: row[col] = 0.0
    df = pd.DataFrame([row])[feature_names]
    return scaler.transform(df) if scaler is not None else df.values


def risk(prob):
    if prob >= 0.8: return 'CRITICAL', '#ef4444'
    if prob >= 0.6: return 'HIGH',     '#f97316'
    if prob >= 0.4: return 'MEDIUM',   '#eab308'
    if prob >= 0.2: return 'LOW',      '#84cc16'
    return              'SAFE',        '#22c55e'

# ─────────────────────────────────────────────────────────────
#  Routes — Auth
# ─────────────────────────────────────────────────────────────
@app.route('/login')
def login_page():
    if session.get('logged_in'):
        return redirect(url_for('home'))
    if not os.path.exists(USERS_FILE):
        return render_template('login.html', error="No admin registered. Run setup_admin.py first.")
    return render_template('login.html', error=None)


@app.route('/auth/verify_face', methods=['POST'])
def auth_verify_face():
    """Step 1 — check face only, return token for step 2."""
    data     = request.json or {}
    username = data.get('username', '').strip()
    image    = data.get('image', '')

    if not username or not image:
        return jsonify({'success': False, 'error': 'Missing username or image'}), 400

    users = load_users()
    if username not in users:
        return jsonify({'success': False, 'error': 'User not found'}), 401

    if verify_face(username, image):
        # Store a temporary face-verified flag in session
        session['face_verified_for'] = username
        return jsonify({'success': True})
    else:
        session.pop('face_verified_for', None)
        return jsonify({'success': False, 'error': 'Face not recognised. Try again.'})


@app.route('/auth/login', methods=['POST'])
def auth_login():
    """Step 2 — verify password, then grant session if face was already verified."""
    data     = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    # Face must have been verified in this session first
    if session.get('face_verified_for') != username:
        return jsonify({'success': False, 'error': 'Face verification required first'}), 401

    if verify_password(username, password):
        session.clear()
        session['logged_in'] = True
        session['username']  = username
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Incorrect password'}), 401


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ─────────────────────────────────────────────────────────────
#  Routes — IDS (protected)
# ─────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def home():
    available = {k: MODEL_LABELS[k] for k in MODELS}
    features  = {k: MODELS[k]['feature_names'] for k in MODELS}
    return render_template('index.html',
                           available=available,
                           features=features,
                           username=session.get('username'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data      = request.json if request.is_json else request.form.to_dict()
        model_key = str(data.pop('model_key', 'cic'))

        if model_key not in MODELS:
            return jsonify({'success': False,
                            'error': f'Model "{model_key}" not loaded.'}), 400

        X    = preprocess(data, model_key)
        prob = float(MODELS[model_key]['model'].predict(X, verbose=0)[0][0])
        pred = 'ATTACK' if prob >= 0.5 else 'NORMAL'
        rl, rc = risk(prob)

        return jsonify({
            'success': True, 'model_used': MODEL_LABELS[model_key],
            'prediction': pred, 'probability': round(prob * 100, 2),
            'confidence': round(max(prob, 1 - prob) * 100, 2),
            'risk_level': rl, 'risk_color': rc,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/status')
@login_required
def status():
    return jsonify({k: {'label': MODEL_LABELS.get(k), 'loaded': True,
                        'features': len(MODELS[k]['feature_names'])} for k in MODELS})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  UNIFIED IDS  —  Face + Password Auth")
    print("="*60)
    if not os.path.exists(USERS_FILE):
        print("  ⚠  No admin found — run: python setup_admin.py")
    print(f"  Active models : {[MODEL_LABELS[k] for k in MODELS]}")
    print("  URL           : http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
