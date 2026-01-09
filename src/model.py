"""
CNN Model for FLIPS - 5 layer architecture
Conv32 → Conv64 → Conv128 → FC256 → Softmax100
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(32, 32, 3), num_classes=100):
    """
    Build 5-layer CNN as specified in FLIPS paper.

    Architecture:
    - Conv1: 32 filters, 3x3, ReLU, MaxPool 2x2
    - Conv2: 64 filters, 3x3, ReLU, MaxPool 2x2
    - Conv3: 128 filters, 3x3, ReLU, MaxPool 2x2
    - FC: 256 neurons, ReLU
    - Output: num_classes neurons, Softmax

    Args:
        input_shape: Input image shape (default: 32x32x3 for CIFAR-100)
        num_classes: Number of output classes (default: 100 for CIFAR-100)

    Returns:
        Keras model
    """
    model = keras.Sequential([
        # Layer 1: Conv32
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Layer 2: Conv64
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Layer 3: Conv128
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        # Flatten
        layers.Flatten(name='flatten'),

        # Layer 4: FC256
        layers.Dense(256, activation='relu', name='fc'),

        # Layer 5: Output (Softmax)
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='FLIPS_CNN')

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
