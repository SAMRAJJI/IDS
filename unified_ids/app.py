"""
Unified Intrusion Detection System
Supports: NSL-KDD  |  CIC-IDS2017

Folder layout:
  IDS with DL/
  ├── ids/models/                       ← NSL-KDD (best_model.h5, scaler.pkl, label_encoders.pkl)
  ├── IDS with DL - CIC2017/models/     ← CIC     (best_model.h5, scaler.pkl, feature_columns.pkl)
  └── unified_ids/                      ← THIS APP
      ├── app.py
      └── templates/index.html

Run: python app.py
"""

import os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator
import joblib
import sys
app = Flask(__name__)

# ── Absolute base = "IDS with DL/" ────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KDD_MODELS_DIR = os.path.join(BASE, 'ids', 'models')
CIC_MODELS_DIR = os.path.join(BASE, 'IDS with DL - CIC2017', 'models')

KDD_FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# ─────────────────────────────────────────────────────────────
#  Load both models
# ─────────────────────────────────────────────────────────────
MODELS = {}

# ── NSL-KDD ───────────────────────────────────────────────────
try:
    print("Loading NSL-KDD ...")
    kdd_model_path    = os.path.join(KDD_MODELS_DIR, 'best_model.h5')
    kdd_scaler_path   = os.path.join(KDD_MODELS_DIR, 'scaler.pkl')
    kdd_encoders_path = os.path.join(KDD_MODELS_DIR, 'label_encoders.pkl')
    print(f"  model  : {kdd_model_path}")
    print(f"  scaler : {kdd_scaler_path}")
    print(f"  encoders: {kdd_encoders_path}")

    kdd_model = tf.keras.models.load_model(kdd_model_path, compile=False)
    with open(kdd_scaler_path, 'rb') as f:
        kdd_scaler = pickle.load(f)
    with open(kdd_encoders_path, 'rb') as f:
        kdd_encoders = pickle.load(f)

    MODELS['kdd'] = {
        'model':         kdd_model,
        'scaler':        kdd_scaler,
        'feature_names': KDD_FEATURE_NAMES,
        'encoders':      kdd_encoders,
    }
    print(f"  ✓ NSL-KDD ready  ({len(KDD_FEATURE_NAMES)} features)")

except Exception as e:
    print(f"  ✗ NSL-KDD failed: {e}")

# ── CIC-IDS2017 ───────────────────────────────────────────────
try:
    print("Loading CIC-IDS2017 ...")
    cic_model_path   = os.path.join(CIC_MODELS_DIR, 'best_model.h5')
    cic_scaler_path  = os.path.join(CIC_MODELS_DIR, 'scaler.pkl')
    cic_columns_path = os.path.join(CIC_MODELS_DIR, 'feature_columns.pkl')
    print(f"  model  : {cic_model_path}")
    print(f"  scaler : {cic_scaler_path}")
    print(f"  cols   : {cic_columns_path}")

    cic_model = tf.keras.models.load_model(cic_model_path)

    with open(cic_scaler_path, 'rb') as f:
        cic_scaler = pickle.load(f)

    with open(cic_columns_path, 'rb') as f:
        cic_columns = pickle.load(f)

    if isinstance(cic_columns, dict):
        cic_feature_names = list(cic_columns.keys())
        cic_encoders      = cic_columns
    else:
        cic_feature_names = list(cic_columns)
        cic_encoders      = {}

    MODELS['cic'] = {
        'model':         cic_model,
        'scaler':        cic_scaler,
        'feature_names': cic_feature_names,
        'encoders':      cic_encoders,
    }
    print(f"  ✓ CIC-IDS2017 ready  ({len(cic_feature_names)} features)")

except Exception as e:
    print(f"  ✗ CIC-IDS2017 failed: {e}")

if not MODELS:
    raise RuntimeError("No models loaded — check folder paths above.")

MODEL_LABELS = {'kdd': 'NSL-KDD', 'cic': 'CIC-IDS2017'}

# ─────────────────────────────────────────────────────────────
#  Helpers
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
            try:
                row[col] = float(val)
            except (ValueError, TypeError):
                row[col] = 0.0

    df = pd.DataFrame([row])[feature_names]
    return scaler.transform(df) if scaler is not None else df.values


def risk(prob):
    if prob >= 0.8: return 'CRITICAL', '#ef4444'
    if prob >= 0.6: return 'HIGH',     '#f97316'
    if prob >= 0.4: return 'MEDIUM',   '#eab308'
    if prob >= 0.2: return 'LOW',      '#84cc16'
    return              'SAFE',        '#22c55e'

# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    available = {k: MODEL_LABELS[k] for k in MODELS}
    features  = {k: MODELS[k]['feature_names'] for k in MODELS}
    return render_template('index.html', available=available, features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data      = request.json if request.is_json else request.form.to_dict()
        model_key = str(data.pop('model_key', 'cic'))

        if model_key not in MODELS:
            return jsonify({'success': False,
                            'error': f'Model "{model_key}" not loaded. '
                                     f'Available: {list(MODELS.keys())}'}), 400

        X    = preprocess(data, model_key)
        prob = float(MODELS[model_key]['model'].predict(X, verbose=0)[0][0])
        pred = 'ATTACK' if prob >= 0.5 else 'NORMAL'
        rl, rc = risk(prob)

        return jsonify({
            'success':     True,
            'model_used':  MODEL_LABELS[model_key],
            'prediction':  pred,
            'probability': round(prob * 100, 2),
            'confidence':  round(max(prob, 1 - prob) * 100, 2),
            'risk_level':  rl,
            'risk_color':  rc,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/status')
def status():
    return jsonify({
        k: {'label': MODEL_LABELS.get(k, k), 'loaded': True,
            'features': len(MODELS[k]['feature_names'])}
        for k in MODELS
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  UNIFIED INTRUSION DETECTION SYSTEM")
    print("="*60)
    print(f"   Active : {[MODEL_LABELS[k] for k in MODELS]}")
    print("   URL    : http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
