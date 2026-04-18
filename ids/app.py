"""
Flask Web Interface for Intrusion Detection System
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and preprocessors
print("Loading model and preprocessors...")
model = tf.keras.models.load_model('models/best_model.h5')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print("✓ Model loaded successfully!")

# Feature names
feature_names = [
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

categorical_features = ['protocol_type', 'service', 'flag']

# Get valid categorical values
valid_values = {}
for col in categorical_features:
    valid_values[col] = list(label_encoders[col].classes_)


def validate_and_encode_categorical(feature_name, value):
    """Validate and encode categorical feature"""
    value_str = str(value).strip()
    
    # Try case-insensitive match
    for valid_value in valid_values[feature_name]:
        if value_str.lower() == valid_value.lower():
            return label_encoders[feature_name].transform([valid_value])[0]
    
    # Default to first value if not found
    default = valid_values[feature_name][0]
    return label_encoders[feature_name].transform([default])[0]


def preprocess_input(data):
    """Preprocess user input"""
    # Create feature vector with defaults
    sample = {
        'duration': int(data.get('duration', 0)),
        'protocol_type': data.get('protocol_type', 'tcp'),
        'service': data.get('service', 'http'),
        'flag': data.get('flag', 'SF'),
        'src_bytes': int(data.get('src_bytes', 0)),
        'dst_bytes': int(data.get('dst_bytes', 0)),
        'land': int(data.get('land', 0)),
        'wrong_fragment': int(data.get('wrong_fragment', 0)),
        'urgent': int(data.get('urgent', 0)),
        'hot': int(data.get('hot', 0)),
        'num_failed_logins': int(data.get('num_failed_logins', 0)),
        'logged_in': int(data.get('logged_in', 1)),
        'num_compromised': int(data.get('num_compromised', 0)),
        'root_shell': int(data.get('root_shell', 0)),
        'su_attempted': int(data.get('su_attempted', 0)),
        'num_root': int(data.get('num_root', 0)),
        'num_file_creations': int(data.get('num_file_creations', 0)),
        'num_shells': int(data.get('num_shells', 0)),
        'num_access_files': int(data.get('num_access_files', 0)),
        'num_outbound_cmds': int(data.get('num_outbound_cmds', 0)),
        'is_host_login': int(data.get('is_host_login', 0)),
        'is_guest_login': int(data.get('is_guest_login', 0)),
        'count': int(data.get('count', 2)),
        'srv_count': int(data.get('srv_count', 2)),
        'serror_rate': float(data.get('serror_rate', 0.0)),
        'srv_serror_rate': float(data.get('srv_serror_rate', 0.0)),
        'rerror_rate': float(data.get('rerror_rate', 0.0)),
        'srv_rerror_rate': float(data.get('srv_rerror_rate', 0.0)),
        'same_srv_rate': float(data.get('same_srv_rate', 1.0)),
        'diff_srv_rate': float(data.get('diff_srv_rate', 0.0)),
        'srv_diff_host_rate': float(data.get('srv_diff_host_rate', 0.0)),
        'dst_host_count': int(data.get('dst_host_count', 255)),
        'dst_host_srv_count': int(data.get('dst_host_srv_count', 255)),
        'dst_host_same_srv_rate': float(data.get('dst_host_same_srv_rate', 1.0)),
        'dst_host_diff_srv_rate': float(data.get('dst_host_diff_srv_rate', 0.0)),
        'dst_host_same_src_port_rate': float(data.get('dst_host_same_src_port_rate', 1.0)),
        'dst_host_srv_diff_host_rate': float(data.get('dst_host_srv_diff_host_rate', 0.0)),
        'dst_host_serror_rate': float(data.get('dst_host_serror_rate', 0.0)),
        'dst_host_srv_serror_rate': float(data.get('dst_host_srv_serror_rate', 0.0)),
        'dst_host_rerror_rate': float(data.get('dst_host_rerror_rate', 0.0)),
        'dst_host_srv_rerror_rate': float(data.get('dst_host_srv_rerror_rate', 0.0))
    }
    
    # Create DataFrame
    df = pd.DataFrame([sample])
    
    # Encode categorical features
    for col in categorical_features:
        df[col] = validate_and_encode_categorical(col, df[col].iloc[0])
    
    # Ensure correct order
    df = df[feature_names]
    
    # Scale
    X_scaled = scaler.transform(df)
    
    return X_scaled


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', 
                         protocols=valid_values['protocol_type'],
                         services=valid_values['service'][:20],  # Show first 20 services
                         flags=valid_values['flag'])


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if it's JSON (API request) or form data (web request)
        if request.is_json:
            data = request.json
        else:
            data = request.form
        
        # Preprocess
        X = preprocess_input(data)
        
        # Predict
        probability = float(model.predict(X, verbose=0)[0][0])
        prediction = "ATTACK" if probability >= 0.5 else "NORMAL"
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = 'CRITICAL'
            risk_color = '#d32f2f'
        elif probability >= 0.6:
            risk_level = 'HIGH'
            risk_color = '#f57c00'
        elif probability >= 0.4:
            risk_level = 'MEDIUM'
            risk_color = '#fbc02d'
        elif probability >= 0.2:
            risk_level = 'LOW'
            risk_color = '#689f38'
        else:
            risk_level = 'SAFE'
            risk_color = '#388e3c'
        
        response = {
            'success': True,
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'confidence': round(max(probability, 1-probability) * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color
        }
        
        # If JSON request, return more detailed info
        if request.is_json:
            response['attack_probability'] = round(probability, 4)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/about')
def about():
    """About page with model info"""
    return jsonify({
        'model': 'Attention-Enhanced Deep Neural Network',
        'dataset': 'NSL-KDD',
        'features': len(feature_names),
        'framework': 'TensorFlow/Keras',
        'valid_protocols': valid_values['protocol_type'],
        'valid_flags': valid_values['flag'],
        'available_services': len(valid_values['service'])
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  INTRUSION DETECTION SYSTEM - WEB INTERFACE")
    print("="*60)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to:")
    print("  → http://localhost:5000")
    print("  → http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)