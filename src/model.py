"""
CNN Model for FLIPS - 5 layer architecture
Conv32 → Conv64 → Conv128 → FC256 → Softmax100
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(32, 32, 3), num_classes=100):
    """
    Robust VGG-style CNN for FLIPS.
    Uses LayerNormalization to avoid breaking delta coding pipelines 
    (no non-trainable moving statistics).
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, name='conv1'),
        layers.LayerNormalization(axis=-1, name='ln1'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', name='conv2'),
        layers.LayerNormalization(axis=-1, name='ln2'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', name='conv3'),
        layers.LayerNormalization(axis=-1, name='ln3'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.4),

        # Flatten & Dense Classifier
        layers.Flatten(name='flatten'),
        
        # Increased dense capacity for 100 classes
        layers.Dense(512, name='fc1'), 
        layers.LayerNormalization(axis=-1, name='ln4'),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='FLIPS_CNN_LayerNorm')

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
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
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
