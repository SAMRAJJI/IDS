import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class AttentionLayer(layers.Layer):
    """Self-attention layer"""
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_context'
        )
        
    def call(self, x):
        # Attention mechanism
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1))
        attention_weights = tf.expand_dims(attention_weights, -1)
        weighted_input = x * attention_weights
        return weighted_input, attention_weights


def build_ae_dnn_model(input_dim):
    """
    Build Attention-Enhanced Deep Neural Network
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
    
    model = Model(inputs=inputs, outputs=outputs, name='AE_DNN_IDS')
    
    return model


def build_simple_dnn_model(input_dim):
    """
    Simpler DNN model (fallback option)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ], name='Simple_DNN_IDS')
    
    return model


if __name__ == "__main__":
    # Test model creation
    input_dim = 41  # KDD has 41 features
    model = build_ae_dnn_model(input_dim)
    model.summary()
    
    # Save model architecture visualization
    keras.utils.plot_model(
        model,
        to_file='results/model_architecture.png',
        show_shapes=True,
        show_layer_names=True
    )
    print("\nModel architecture saved to results/model_architecture.png")