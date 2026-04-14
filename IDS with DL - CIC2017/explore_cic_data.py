"""
Explore CIC-IDS2017 Dataset
Run this first to understand your data
"""
import pandas as pd
import os
import glob

def explore_cic_dataset(data_folder='data/raw'):
    """Explore CIC-IDS2017 CSV files"""
    
    print("="*70)
    print("CIC-IDS2017 DATASET EXPLORATION")
    print("="*70)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    
    if not csv_files:
        print("\n❌ No CSV files found in", data_folder)
        print("Please move your CIC-IDS2017 CSV files to data/raw/ folder")
        return
    
    print(f"\n📁 Found {len(csv_files)} CSV files:\n")
    
    total_rows = 0
    all_labels = set()
    
    for i, file in enumerate(csv_files, 1):
        filename = os.path.basename(file)
        print(f"{i}. {filename}")
        
        try:
            # Read first few rows to get info
            df = pd.read_csv(file, nrows=1000)
            
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            total_file_rows = sum(1 for _ in open(file, encoding='utf-8', errors='ignore')) - 1
            
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Rows: {total_file_rows:,}")
            print(f"   Columns: {len(df.columns)}")
            
            # Check label column
            label_col = df.columns[-1]  # Usually last column
            print(f"   Label column: '{label_col}'")
            
            # Get unique labels in this file
            df_full = pd.read_csv(file)
            labels = df_full[label_col].unique()
            print(f"   Labels: {', '.join(map(str, labels[:5]))}" + 
                  ("..." if len(labels) > 5 else ""))
            
            all_labels.update(labels)
            total_rows += total_file_rows
            print()
            
        except Exception as e:
            print(f"   ⚠️ Error reading file: {e}\n")
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files: {len(csv_files)}")
    print(f"Total rows: {total_rows:,}")
    print(f"\nAll unique labels found ({len(all_labels)}):")
    for label in sorted(all_labels):
        print(f"  - {label}")
    print("="*70)
    
    # Show feature names from first file
    if csv_files:
        print("\n📊 FEATURES (from first file):")
        print("="*70)
        df = pd.read_csv(csv_files[0], nrows=5)
        print(f"Total features: {len(df.columns)}\n")
        
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        print("="*70)
    
    return csv_files, all_labels


if __name__ == "__main__":
    csv_files, labels = explore_cic_dataset('data/raw')
    
    if csv_files:
        print("\n✅ Dataset exploration complete!")
        print("\nNext steps:")
        print("1. Run preprocessing_cic.py to prepare the data")
        print("2. Train the model with train.py")
    else:
        print("\n⚠️ Please add CIC-IDS2017 CSV files to data/raw/ folder first")