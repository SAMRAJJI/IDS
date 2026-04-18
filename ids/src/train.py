import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import build_ae_dnn_model, build_simple_dnn_model
import json
from datetime import datetime

class IDSTrainer:
    def __init__(self, model_type='ae_dnn'):
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"\nClass weights: {class_weight_dict}")
        return class_weight_dict
    
    def build_model(self, input_dim):
        """Build model based on type"""
        if self.model_type == 'ae_dnn':
            self.model = build_ae_dnn_model(input_dim)
        else:
            self.model = build_simple_dnn_model(input_dim)
        
        # Compile model
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
        
        print("\nModel compiled successfully!")
        self.model.summary()
        
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        return callbacks
    
    def train(self, epochs=100, batch_size=128):
        """Train the model"""
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_data()
        
        # Build model
        self.build_model(X_train.shape[1])
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Save final model
        self.model.save('models/final_model.h5')
        print("Model saved to models/final_model.h5")
        
        # Save training history
        with open('results/training_history.json', 'w') as f:
            json.dump({k: [float(v) for v in val] for k, val in self.history.history.items()}, f)
        
        return self.history
    
    def plot_training_history(self):
        """Plot training metrics"""
        if self.history is None:
            print("No training history available!")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history['precision'], label='Train Precision')
        axes[1, 0].plot(history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history['recall'], label='Train Recall')
        axes[1, 1].plot(history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300)
        print("Training history plot saved to results/training_history.png")
        plt.show()


if __name__ == "__main__":
    trainer = IDSTrainer(model_type='ae_dnn')
    history = trainer.train(epochs=50, batch_size=128)
    trainer.plot_training_history()