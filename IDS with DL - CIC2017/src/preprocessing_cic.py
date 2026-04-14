"""
CIC-IDS2017 Dataset Preprocessing
Handles 2.8M samples with 78 features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import glob
import os

class CICPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_all_files(self, data_folder='data/raw'):
        """Load and combine all CIC CSV files"""
        print("="*70)
        print("LOADING CIC-IDS2017 DATASET")
        print("="*70)
        
        csv_files = sorted(glob.glob(os.path.join(data_folder, '*.csv')))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_folder}")
        
        print(f"\nFound {len(csv_files)} files")
        
        all_dataframes = []
        
        for i, file in enumerate(csv_files, 1):
            filename = os.path.basename(file)
            print(f"\n{i}. Loading {filename}...")
            
            try:
                # Read CSV
                df = pd.read_csv(file)
                print(f"   Loaded {len(df):,} rows")
                
                all_dataframes.append(df)
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        
        # Combine all files
        print("\n" + "-"*70)
        print("Combining all files...")
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total rows: {len(df_combined):,}")
        print(f"Total columns: {len(df_combined.columns)}")
        
        return df_combined
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print("\n" + "="*70)
        print("CLEANING DATA")
        print("="*70)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        print("\n1. Handling missing values...")
        print(f"   Missing values before: {df.isnull().sum().sum()}")
        df = df.fillna(0)
        print(f"   Missing values after: {df.isnull().sum().sum()}")
        
        # Handle infinity values
        print("\n2. Handling infinity values...")
        # Replace inf with large numbers
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        print("\n3. Removing duplicates...")
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        print(f"   Removed {before - after:,} duplicates")
        
        return df
    
    def create_binary_labels(self, df):
        """Convert to binary: BENIGN vs ATTACK"""
        print("\n" + "="*70)
        print("CREATING BINARY LABELS")
        print("="*70)
        
        # Create binary label
        df['binary_label'] = df['Label'].apply(
            lambda x: 0 if str(x).strip() == 'BENIGN' else 1
        )
        
        print("\nLabel distribution:")
        print(df['Label'].value_counts())
        
        print("\nBinary label distribution:")
        print(f"BENIGN (0): {(df['binary_label'] == 0).sum():,}")
        print(f"ATTACK (1): {(df['binary_label'] == 1).sum():,}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix"""
        print("\n" + "="*70)
        print("PREPARING FEATURES")
        print("="*70)
        
        # Separate features and labels
        X = df.drop(['Label', 'binary_label'], axis=1)
        y = df['binary_label']
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        print(f"\nFeatures: {len(self.feature_columns)}")
        print(f"Samples: {len(X):,}")
        
        # Check for non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"\n⚠️ Non-numeric columns found: {list(non_numeric)}")
            print("Converting to numeric...")
            for col in non_numeric:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split into train/val/test"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"  - BENIGN: {(y_train == 0).sum():,}")
        print(f"  - ATTACK: {(y_train == 1).sum():,}")
        
        print(f"\nValidation set: {X_val.shape}")
        print(f"  - BENIGN: {(y_val == 0).sum():,}")
        print(f"  - ATTACK: {(y_val == 1).sum():,}")
        
        print(f"\nTest set: {X_test.shape}")
        print(f"  - BENIGN: {(y_test == 0).sum():,}")
        print(f"  - ATTACK: {(y_test == 1).sum():,}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Standardize features"""
        print("\n" + "="*70)
        print("SCALING FEATURES")
        print("="*70)
        
        print("\nFitting scaler on training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Transforming validation data...")
        X_val_scaled = self.scaler.transform(X_val)
        
        print("Transforming test data...")
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Scaling complete")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_preprocessors(self, output_dir='models'):
        """Save scaler and feature names"""
        print("\n" + "="*70)
        print("SAVING PREPROCESSORS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to {output_dir}/scaler.pkl")
        
        # Save feature columns
        with open(f'{output_dir}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"✓ Saved feature columns to {output_dir}/feature_columns.pkl")
    
    def preprocess_full_pipeline(self):
        """Complete preprocessing pipeline"""
        print("\n" + "🛡️ "*35)
        print("CIC-IDS2017 PREPROCESSING PIPELINE")
        print("🛡️ "*35 + "\n")
        
        # Step 1: Load all files
        df = self.load_all_files('data/raw')
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Create binary labels
        df = self.create_binary_labels(df)
        
        # Step 4: Prepare features
        X, y = self.prepare_features(df)
        
        # Step 5: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 6: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Step 7: Save preprocessed data
        print("\n" + "="*70)
        print("SAVING PREPROCESSED DATA")
        print("="*70)
        
        os.makedirs('data/processed', exist_ok=True)
        
        np.save('data/processed/X_train.npy', X_train_scaled)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/X_val.npy', X_val_scaled)
        np.save('data/processed/y_val.npy', y_val)
        np.save('data/processed/X_test.npy', X_test_scaled)
        np.save('data/processed/y_test.npy', y_test)
        
        print("\n✓ Saved processed data to data/processed/")
        print("  - X_train.npy, y_train.npy")
        print("  - X_val.npy, y_val.npy")
        print("  - X_test.npy, y_test.npy")
        
        # Step 8: Save preprocessors
        self.save_preprocessors()
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nDataset ready for training:")
        print(f"  - Features: {len(self.feature_columns)}")
        print(f"  - Training samples: {len(y_train):,}")
        print(f"  - Validation samples: {len(y_val):,}")
        print(f"  - Test samples: {len(y_test):,}")
        print("\nNext step: Run training")
        print("  python src/train.py")
        print("="*70)
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    preprocessor = CICPreprocessor()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.preprocess_full_pipeline()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.2f} minutes")