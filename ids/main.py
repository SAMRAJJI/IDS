"""
Main pipeline for Deep Learning-based Intrusion Detection System
"""
import argparse
import sys
from src.preprocessing import KDDPreprocessor
from src.train import IDSTrainer
from src.evaluate import IDSEvaluator
import numpy as np

def preprocess_data(data_path):
    """Preprocess KDD dataset"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = KDDPreprocessor()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.preprocess(data_path)
    
    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("\n✓ Preprocessing completed!")

def train_model(model_type='ae_dnn', epochs=50):
    """Train IDS model"""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    trainer = IDSTrainer(model_type=model_type)
    history = trainer.train(epochs=epochs, batch_size=128)
    trainer.plot_training_history()
    
    print("\n✓ Training completed!")

def evaluate_model(model_path='models/best_model.h5'):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    evaluator = IDSEvaluator(model_path)
    y_pred, y_pred_proba = evaluator.evaluate(threshold=0.5)
    
    print("\n✓ Evaluation completed!")

def run_full_pipeline(data_path, model_type='ae_dnn', epochs=50):
    """Run complete pipeline"""
    print("\n" + "="*60)
    print("RUNNING FULL IDS PIPELINE")
    print("="*60)
    
    # Step 1: Preprocess
    preprocess_data(data_path)
    
    # Step 2: Train
    train_model(model_type, epochs)
    
    # Step 3: Evaluate
    evaluate_model()
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck the following folders for results:")
    print("  - models/     : Saved models")
    print("  - results/    : Plots and metrics")
    print("  - logs/       : Training logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning IDS')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['preprocess', 'train', 'evaluate', 'full'],
                       help='Pipeline mode')
    parser.add_argument('--data', type=str, default='data/kddcup.data_10_percent',
                       help='Path to KDD dataset')
    parser.add_argument('--model', type=str, default='ae_dnn',
                       choices=['ae_dnn', 'simple_dnn'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_pipeline(args.data, args.model, args.epochs)
    elif args.mode == 'preprocess':
        preprocess_data(args.data)
    elif args.mode == 'train':
        train_model(args.model, args.epochs)
    elif args.mode == 'evaluate':
        evaluate_model()