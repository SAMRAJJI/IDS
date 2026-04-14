"""
Deep Learning Model for CIC-IDS2017
Updated for 78 input features
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_ae_dnn_model(input_dim=78):
    """
    Build Attention-Enhanced DNN for CIC-IDS2017
    Input: 78 features (instead of 41 for KDD)
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Dense Block 1
    x = layers.Dense(256, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Residual connection
    x_res = layers.Dense(256)(inputs)
    x = layers.Add()([x, x_res])
    
    # Dense Block 2
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Attention Layer
    x_reshaped = layers.Reshape((1, 128))(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        dropout=0.1
    )(x_reshaped, x_reshaped)
    attention_output = layers.LayerNormalization()(attention_output)
    x = layers.Add()([attention_output, x_reshaped])
    x = layers.Flatten()(x)
    
    # Dense Block 3
    x = layers.Dense(64, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AE_DNN_CIC')
    
    return model


if __name__ == "__main__":
    # Test model creation
    input_dim = 78  # CIC-IDS2017 has 78 features
    model = build_ae_dnn_model(input_dim)
    model.summary()
    
    print(f"\n✓ Model created for {input_dim} input features")
    print(f"✓ Total parameters: {model.count_params():,}")