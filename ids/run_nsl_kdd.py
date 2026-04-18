"""
Run IDS with NSL-KDD Dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# NSL-KDD column names
columns = [
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
    'dst_host_srv_rerror_rate', 'label', 'difficulty'  # NSL-KDD has extra 'difficulty' column
]

print("="*60)
print("PREPROCESSING NSL-KDD DATASET")
print("="*60)

# Load training data
print("\n1. Loading training data...")
train_file = 'data/nsl-kdd/KDDTrain+_20Percent.txt'  # Using 20% for faster processing
df_train = pd.read_csv(train_file, names=columns, header=None)
print(f"Training data shape: {df_train.shape}")

# Load test data
print("\n2. Loading test data...")
test_file = 'data/nsl-kdd/KDDTest+.txt'
df_test = pd.read_csv(test_file, names=columns, header=None)
print(f"Test data shape: {df_test.shape}")

# Create binary labels
print("\n3. Creating binary labels (0=Normal, 1=Attack)...")
df_train['binary_label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['binary_label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

print(f"Training - Normal: {(df_train['binary_label']==0).sum()}, Attack: {(df_train['binary_label']==1).sum()}")
print(f"Test - Normal: {(df_test['binary_label']==0).sum()}, Attack: {(df_test['binary_label']==1).sum()}")

# Drop unnecessary columns
print("\n4. Preparing features...")
df_train = df_train.drop(['label', 'difficulty'], axis=1)
df_test = df_test.drop(['label', 'difficulty'], axis=1)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

print("\n5. Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to handle all categories
    combined = pd.concat([df_train[col], df_test[col]])
    le.fit(combined.astype(str))
    
    df_train[col] = le.transform(df_train[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))
    label_encoders[col] = le

# Separate features and labels
X_train_full = df_train.drop('binary_label', axis=1)
y_train_full = df_train['binary_label']
X_test = df_test.drop('binary_label', axis=1)
y_test = df_test['binary_label']

# Create validation set from training
print("\n6. Creating validation set...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
print("\n7. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save processed data
print("\n8. Saving processed data...")
np.save('data/X_train.npy', X_train_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/X_val.npy', X_val_scaled)
np.save('data/y_val.npy', y_val)
np.save('data/X_test.npy', X_test_scaled)
np.save('data/y_test.npy', y_test)

# Save preprocessors
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*60)
print("✓ PREPROCESSING COMPLETED!")
print("="*60)
print("\nProcessed files saved in data/ folder:")
print("  - X_train.npy, y_train.npy")
print("  - X_val.npy, y_val.npy")
print("  - X_test.npy, y_test.npy")
print("\nPreprocessors saved in models/ folder:")
print("  - label_encoders.pkl")
print("  - scaler.pkl")
print("\n" + "="*60)
print("NEXT STEP: Run training")
print("="*60)
print("python src/train.py")