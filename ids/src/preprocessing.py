import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class KDDPreprocessor:
    def __init__(self):
        # KDD Cup 99 column names
        self.columns = [
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
            'dst_host_srv_rerror_rate', 'label'
        ]
        
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load KDD dataset"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, names=self.columns, header=None)
        print(f"Data loaded. Shape: {df.shape}")
        return df
    
    def create_binary_labels(self, df):
        """Convert to binary classification: normal vs attack"""
        print("\nCreating binary labels...")
        df['binary_label'] = df['label'].apply(
            lambda x: 0 if x == 'normal.' else 1
        )
        
        # Show class distribution
        print("\nClass Distribution:")
        print(df['binary_label'].value_counts())
        print(f"Normal: {(df['binary_label']==0).sum()}")
        print(f"Attack: {(df['binary_label']==1).sum()}")
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical features"""
        print("\nEncoding categorical features...")
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """Standardize numerical features"""
        print("\nScaling features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def preprocess(self, filepath, test_size=0.2, val_size=0.1):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        # Create binary labels
        df = self.create_binary_labels(df)
        
        # Encode categorical
        df = self.encode_categorical(df, fit=True)
        
        # Separate features and labels
        X = df.drop(['label', 'binary_label'], axis=1)
        y = df['binary_label']
        
        # Split data
        print("\nSplitting data...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Val set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_val_scaled = self.scale_features(X_val, fit=False)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        # Save preprocessors
        self.save_preprocessors()
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def save_preprocessors(self, path='models/'):
        """Save encoders and scaler"""
        with open(f'{path}label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(f'{path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("\nPreprocessors saved!")


if __name__ == "__main__":
    preprocessor = KDDPreprocessor()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.preprocess(
        'data/kddcup.data_10_percent'
    )
    
    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    print("\nProcessed data saved!")