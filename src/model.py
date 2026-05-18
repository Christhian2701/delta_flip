"""
CNN Model for FLIPS - 5 layer architecture
Conv32 → Conv64 → Conv128 → FC256 → Softmax100

using batch normalization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def build_cnn(input_shape=(32, 32, 3), num_classes=100):
    """
    Build 5-layer CNN as specified in FLIPS paper.

    Architecture: # mudança


    Args:
        input_shape: Input image shape (default: 32x32x3 for CIFAR-100)
        num_classes: Number of output classes (default: 100 for CIFAR-100)

    Returns:
        Keras model
    """
    reg = regularizers.l2(1e-4)
    init = 'he_normal'

    model = keras.Sequential([
        # Layer 1: Conv64 (Widened)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=init, kernel_regularizer=reg,
                     input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.2, name='drop1'), # 0 parameters, won't affect deltas!

        # Layer 2: Conv128 (Widened)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                     kernel_initializer=init, kernel_regularizer=reg, name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.3, name='drop2'),

        # Layer 3: Conv256 (Widened)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                     kernel_initializer=init, kernel_regularizer=reg, name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.4, name='drop3'),

        # Flatten
        layers.Flatten(name='flatten'),

        # Layer 4: FC512 (Widened to handle more features)
        layers.Dense(512, activation='relu', 
                    kernel_initializer=init, kernel_regularizer=reg, name='fc'),
        layers.Dropout(0.5, name='drop_fc'),

        # Layer 5: Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='FLIPS_CNN_WIDER')

    return model



def compile_model(model, learning_rate=0.01):
    """
    Compile model with optimizer and loss function.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for SGD optimizer

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_layer_names(model):
    """Get names of trainable layers (excluding pooling/flatten)."""
    layer_names = []
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:  # Has trainable weights
            layer_names.append(layer.name)
    return layer_names


def get_model_size(model):
    """Calculate model size in bytes."""
    total_size = 0
    for layer in model.layers:
        for weight in layer.get_weights():
            total_size += weight.nbytes
    return total_size


if __name__ == "__main__":
    # Test model building
    print("Building FLIPS CNN model...")
    model = build_cnn()
    model = compile_model(model)

    print("\nModel Summary:")
    model.summary()

    print(f"\nTrainable layers: {get_layer_names(model)}")
    print(f"Model size: {get_model_size(model) / 1024:.2f} KB")
    print(f"Total parameters: {model.count_params():,}")
