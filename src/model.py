"""
VGG Model for FLIPS - Translated from PyTorch PFLlib
Supports VGG11, VGG13, VGG16, VGG19
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# VGG Architecture Configurations
# Integers represent Conv2D filter counts. 'M' represents MaxPooling.
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def build_cnn(input_shape=(32, 32, 3), num_classes= 100, vgg_name='VGG11'):
    """
    Build VGG network mirroring the PyTorch implementation.
    
    Args:
        input_shape: Input image shape (default: 32x32x3 for CIFAR-10)
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        vgg_name: String identifier for VGG version from cfg dict
        
    Returns:
        Keras model
    """
    model = keras.Sequential(name=vgg_name)
    
    reg = regularizers.l2(1e-4)
    init = 'he_normal'
    
    # --- 1. FEATURE EXTRACTOR ---
    is_first_layer = True
    
    for x in cfg[vgg_name]:
        if x == 'M':
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            if is_first_layer:
                model.add(layers.Conv2D(x, kernel_size=(3, 3), padding='same', 
                                      kernel_initializer=init, kernel_regularizer=reg, 
                                      input_shape=input_shape))
                is_first_layer = False
            else:
                model.add(layers.Conv2D(x, kernel_size=(3, 3), padding='same', 
                                      kernel_initializer=init, kernel_regularizer=reg))
            
            # Exact mirror of PyTorch's nn.BatchNorm2d(x)
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            
    # PyTorch's nn.AvgPool2d(1, 1) on a 32x32 image with 5 MaxPools results 
    # in a 1x1 map. Flatten() is mathematically identical here.
    model.add(layers.Flatten(name='flatten'))
    
    # --- 2. CLASSIFIER ---
    # Mirrors the nn.Sequential logic in the professor's snippet
    model.add(layers.Dense(512, kernel_initializer=init, kernel_regularizer=reg))
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(512, kernel_initializer=init, kernel_regularizer=reg))
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def compile_model(model, learning_rate=0.01):
    """
    Compile model with SGD + Nesterov Momentum.
    """
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=0.9, 
            nesterov=True
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_layer_names(model):
    """Get names of trainable layers."""
    layer_names = []
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
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
    print("Building FLIPS VGG model...")
    model = build_cnn(num_classes=100, vgg_name='VGG11') 
    
    model = compile_model(model)

    print("\nModel Summary:")
    model.summary()

    print(f"\nModel size: {get_model_size(model) / (1024 * 1024):.2f} MB")
    print(f"Total parameters: {model.count_params():,}")