"""
Training script for CIC-IDS2017 (Optimized)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from model import build_ae_dnn_model
import json
from datetime import datetime
import os

# ─────────────────────────────────────────
#  SPEED CONFIG  ← change these as needed
# ─────────────────────────────────────────
BATCH_SIZE   = 2048   # was 256  → ~8x fewer steps per epoch
SAMPLE_FRAC  = 0.25   # use 25% of train data  (set 1.0 for full data)
EPOCHS       = 20     # was 50   → EarlyStopping will cut it shorter
# ─────────────────────────────────────────

class IDSTrainer:
    def __init__(self, model_type='ae_dnn'):
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load preprocessed CIC data"""
        print("Loading preprocessed data...")
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_val   = np.load('data/processed/X_val.npy')
        y_val   = np.load('data/processed/y_val.npy')
        X_test  = np.load('data/processed/X_test.npy')
        y_test  = np.load('data/processed/y_test.npy')
        
        print(f"Full  → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # ── Subsample training set ──────────────────────────────────────
        if SAMPLE_FRAC < 1.0:
            n = int(len(X_train) * SAMPLE_FRAC)
            idx = np.random.default_rng(42).choice(len(X_train), n, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            print(f"Sampled → Train: {X_train.shape}  ({SAMPLE_FRAC*100:.0f}% of data)")
        # ───────────────────────────────────────────────────────────────

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights"""
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"\nClass weights: {class_weight_dict}")
        return class_weight_dict
    
    def build_model(self, input_dim):
        """Build model"""
        self.model = build_ae_dnn_model(input_dim)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("\n✓ Model compiled!")
        self.model.summary()
        
    def get_callbacks(self):
        """Training callbacks"""
        os.makedirs('logs/fit', exist_ok=True)
        
        callbacks = [
            # ── Stop early if val_auc stops improving ──────────────────
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,            # was 10 on val_loss
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            # ── Reduce LR faster ──────────────────────────────────────
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,            # was 5
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the model"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_data()
        
        self.build_model(X_train.shape[1])
        
        class_weights = self.calculate_class_weights(y_train)
        
        callbacks = self.get_callbacks()

        # ── Estimated steps info ──────────────────────────────────────
        steps = int(np.ceil(len(X_train) / batch_size))
        print(f"\nSteps/epoch : {steps}  (batch={batch_size}, samples={len(X_train):,})")
        # ─────────────────────────────────────────────────────────────
        
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save('models/final_model.h5')
        print("\n✓ Training complete!")
        
        os.makedirs('results', exist_ok=True)
        with open('results/training_history.json', 'w') as f:
            json.dump({k: [float(v) for v in val] 
                      for k, val in self.history.history.items()}, f)
        
        return self.history
    
    def plot_training_history(self):
        """Plot metrics"""
        if self.history is None:
            print("No history to plot")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(history['loss'],     label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['accuracy'],     label='Train')
        axes[0, 1].plot(history['val_accuracy'], label='Val')
        axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
        
        axes[1, 0].plot(history['precision'],     label='Train')
        axes[1, 0].plot(history['val_precision'], label='Val')
        axes[1, 0].set_title('Precision'); axes[1, 0].legend(); axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['recall'],     label='Train')
        axes[1, 1].plot(history['val_recall'], label='Val')
        axes[1, 1].set_title('Recall'); axes[1, 1].legend(); axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300)
        print("✓ Training plots saved")
        plt.show()


if __name__ == "__main__":
    trainer = IDSTrainer()
    history = trainer.train()
    trainer.plot_training_history()