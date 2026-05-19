"""
FLIPS Client implementation.

Updated to handle tf.data.Dataset with built-in augmentation and batching.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gzip
import pickle
import copy
import math
import os
import csv

class FLIPSClient:
    """
    FLIPS Client - handles local training and model updates.
    """

    def __init__(self, client_id, data, model, config):
        """
        Initialize FLIPS client.

        Args:
            client_id: Unique client identifier
            data: Dict with 'train_ds', 'val_ds', 'num_samples'
            model: Keras model (will be cloned)
            config: Configuration dict
        """
        self.client_id = client_id
        
        # NEW: Bind the tf.data.Dataset objects directly
        self.train_ds = data['train_ds']
        self.val_ds = data['val_ds']
        self.num_samples = data['num_samples']

        # tracking for compression analysis
        self.round_track = {
            'max_rounds' : config.get('num_rounds', 10),
            'algorithm': config.get('algorithm'),
            'current_round': 0
        }

        # Clone model for this client
        self.old_model = None 
        self.model = model
        self.model.set_weights(model.get_weights())
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=config['learning_rate'],
            momentum=0.9,
            nesterov=True),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Configuration
        self.config = config
        self.local_epochs = config['local_epochs']
        # Note: batch_size is no longer needed here since train_ds is already batched

        # Metrics
        self.local_accuracy = 0.0
        self.local_loss = 0.0
        
        # Phase 3: Mobility context & Metrics
        self.contact_time = 1.0 
        self.rssi_norm = 1.0
        self.dropout_count = 0
        self.training_time = np.random.uniform(0.5, 2.0) 
        self.device_density = 0.0 

    def get_context_factor(self):
        """
        Compute context factor omega_k (Eq. 305).
        """
        g1 = self.config.get('gamma1_rssi', 0.33)
        g2 = self.config.get('gamma2_dropout', 0.33)
        g3 = self.config.get('gamma3_contact', 0.33)
        
        term1 = g1 * self.rssi_norm
        term2 = g2 * (1.0 / (1.0 + self.dropout_count))
        term3 = g3 * self.contact_time
        
        return min(1.0, term1 + term2 + term3)
    
    def _train_fedprox(self, global_weights):
        """
        FedProx Local Training: L(w) + (mu/2) * ||w - w^t||^2
        """
        mu = self.config.get('mu', 0.01)
        optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'],
        momentum=0.9,
        nesterov=True)
        
        global_kernel_weights = [tf.convert_to_tensor(w) for w in global_weights]
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        for epoch in range(self.local_epochs):
            # NEW: Just iterate directly over the already batched/shuffled train_ds
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                    
                    proximal_term = 0.0
                    for i, w in enumerate(self.model.trainable_variables):
                        if i < len(global_kernel_weights):
                            proximal_term += tf.nn.l2_loss(w - global_kernel_weights[i])
                        
                    loss_value += (mu / 2.0) * proximal_term
                    
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
        # NEW: Evaluate using the val_ds
        val_loss, val_acc = self.model.evaluate(self.val_ds, verbose=0)
        return val_loss, val_acc

    def train_local(self, global_weights, active_indices=None, round_num=0):
        """
        Perform local training and FLIPS specific operations.
        """ 
        self.old_model = [np.copy(w) for w in global_weights]
        self.model.set_weights(global_weights)

        # FedProx Implementation
        if self.config.get('algorithm') == 'fedprox':
            if 'mu' not in self.config:
                self.config['mu'] = 0.01 
            self.local_loss, self.local_accuracy = self._train_fedprox(global_weights)
            
            local_weights = self.model.get_weights()
            if active_indices is not None:
                local_weights = [w if i in active_indices else None for i, w in enumerate(local_weights)]

            try:
                _, deltas_dictionary = self.quantize_and_compress(round_num)
            except Exception as e:
                print(f"Error during quantization/compression for client {self.client_id}: {e}")
                deltas_dictionary = None

            return local_weights, self.num_samples, {}, 0, deltas_dictionary
            
        # Standard FedAvg / FLIPS training
        # NEW: Cleaned up double fit call, strictly using the Dataset objects
        history = self.model.fit(
            self.train_ds, 
            validation_data=self.val_ds,
            verbose=0,
            epochs=self.local_epochs
        )

        self.local_loss = history.history['loss'][-1]
        self.local_accuracy = history.history['val_accuracy'][-1]

        # FedAvg Baseline check
        algorithm = self.config.get('algorithm', 'flips')
        if algorithm in ['fedavg', 'fedlama']:
            local_weights = self.model.get_weights()
            
            if active_indices is not None:
                local_weights = [w if i in active_indices else None for i, w in enumerate(local_weights)]
            
            try:
                _, deltas_dictionary = self.quantize_and_compress(round_num)
            except Exception as e:
                print(f"Error during quantization/compression for client {self.client_id}: {e}")
                deltas_dictionary = None

            return local_weights, self.num_samples, {}, 0, deltas_dictionary 

        # Phase 2 logic (FLIPS)
        omega = self.get_context_factor()
        raw_importance = self.compute_shap_importance()
        
        importance_scores = {k: v * omega for k, v in raw_importance.items()}

        if importance_scores:
            max_imp = max(importance_scores.values())
            if max_imp > 0:
                importance_scores = {k: v / max_imp for k, v in importance_scores.items()}
        
        pruning_ratio = self.prune_model(importance_scores, self.contact_time)
        compressed_size, deltas_dictionary = self.quantize_and_compress(round_num)
        local_weights = self.model.get_weights()

        return local_weights, self.num_samples, importance_scores, compressed_size, deltas_dictionary

    def compute_shap_importance(self):
        """
        Compute SHAP-based layer importance using Deep SHAP.
        """
        layer_importance = {}

        # NEW: Since data is in tf.data.Dataset (which is not subscriptable like a list),
        # we grab exactly one batch of validation data to approximate SHAP. 
        # If val_ds is somehow empty, we fallback to a batch from train_ds.
        try:
            sample_X, sample_y = next(iter(self.val_ds))
        except StopIteration:
            sample_X, sample_y = next(iter(self.train_ds))

        try:
            with tf.GradientTape(persistent=True) as tape:
                predictions = self.model(sample_X, training=False)
                loss = tf.keras.losses.sparse_categorical_crossentropy(sample_y, predictions)
                loss = tf.reduce_mean(loss)

            for layer in self.model.layers:
                if not layer.trainable_weights:
                    continue

                kernel = layer.trainable_weights[0]
                grad = tape.gradient(loss, kernel)

                if grad is None:
                    continue

                importance = tf.reduce_mean(tf.abs(grad * kernel))
                layer_importance[layer.name] = float(importance.numpy())

            del tape 

        except Exception as e:
            print(f"Warning: Gradient-based SHAP computation failed: {e}")
            layer_outputs = self._get_layer_outputs(sample_X)

            for layer_name, output in layer_outputs.items():
                if output is not None:
                    importance = float(np.var(output))
                    layer_importance[layer_name] = importance

        if not layer_importance:
            print("Warning: Using weight magnitude fallback for SHAP importance")
            for layer in self.model.layers:
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    w = layer.get_weights()[0]
                    layer_importance[layer.name] = float(np.mean(np.abs(w)))

        return layer_importance

    def _get_layer_outputs(self, input_data):
        layer_outputs = {}
        for i, layer in enumerate(self.model.layers):
            if not layer.trainable_weights:
                continue
            try:
                intermediate_model = keras.Model(inputs=self.model.input, outputs=layer.output)
                activations = intermediate_model.predict(input_data, verbose=0, batch_size=32)
                layer_outputs[layer.name] = activations
            except Exception as e:
                continue
        return layer_outputs

    def prune_model(self, importance_scores, contact_time):
        theta_base_val = self.config.get('theta_base', 0.01)
        alpha = self.config.get('alpha_contact', 0.5) 
        
        adjusted_base = theta_base_val * (1.0 + alpha * contact_time)
        max_importance = max(importance_scores.values()) if importance_scores else 1.0
        
        total_params = 0
        pruned_params = 0
        
        for layer in self.model.layers:
            if layer.name not in importance_scores:
                continue
                
            importance = importance_scores[layer.name]
            threshold = adjusted_base * (1.0 - importance / max_importance)
            
            weights = layer.get_weights()
            if not weights: continue
            
            w = weights[0]
            mask = np.abs(w) > threshold
            w_pruned = w * mask
            
            pruned_count = w.size - np.sum(mask)
            total_params += w.size
            pruned_params += pruned_count
            
            weights[0] = w_pruned
            layer.set_weights(weights)
            
        return pruned_params / max(1, total_params)

    def quantize_and_compress(self, round_num=0):
        weights = self.model.get_weights()
        old_weights = self.old_model if self.old_model is not None else [np.zeros_like(w) for w in weights]

        deltas = self.get_deltas(old_weights, weights)
        deltas_dictionary = self.get_flat(deltas)
        deltas_dictionary['vector'], deltas_dictionary['scale'] = self.uniform_quantization(deltas_dictionary['vector'])
        encoded_deltas = self.rle_encoding(deltas_dictionary['vector'])
        deltas_dictionary['vector'] = encoded_deltas

        delta_serialized = pickle.dumps(deltas_dictionary)
        delta_compressed = gzip.compress(delta_serialized)
        delta_size_bytes = len(delta_compressed)
        
        quantized_weights = []
        for w in weights:
            quantized_weights.append(w.astype(np.float16))
            
        serialized = pickle.dumps(quantized_weights)
        compressed = gzip.compress(serialized)
        original_size_bytes = len(compressed)
         
        self.comparison(original_size_bytes, delta_size_bytes, round_num)
        
        return len(compressed), deltas_dictionary

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        # Note: Server usually passes raw global test sets here (numpy arrays)
        # So this function is safe to leave as-is for server-side evaluation
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    def get_deltas(self, global_weights, new_weights):
        return [new_w - old_w for new_w, old_w in zip(new_weights, global_weights)]

    def get_flat(self, weights):
        tensors = []
        metadata = {}

        for index, layer in enumerate(weights):
            metadata[index] = {
                'shape': layer.shape,
                'size': layer.size
            }
            tensors.append(layer.flatten())

        deltas_flat = {
            'vector': np.concatenate(tensors),
            'metadata': metadata
        }
        return deltas_flat

    def uniform_quantization(self, vector):
        max_abs = np.max(np.abs(vector))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0 # Prevent division by zero
        result = np.clip(np.round(vector / scale), -127, 127).astype(np.int8)
        return result, scale

    def rle_encoding(self, values):
        sentinel = -128
        max_run = 127 
        encoded = []
        i, n = 0, len(values) 

        while i < n:
            if abs(values[i]) == 0:
                sequence_length = 0
                while i < n and abs(values[i]) == 0:
                    sequence_length += 1
                    i += 1
                
                while sequence_length > 0:
                    chunk = min(sequence_length, max_run)
                    encoded.append(sentinel)
                    encoded.append(chunk)
                    sequence_length -= chunk
            else:
                encoded.append(values[i])
                i += 1
        return encoded

    def comparison(self, bytes_original, bytes_delta, round_num):
        if bytes_original > 0:
            reduction_percent = (1 - (bytes_delta / bytes_original)) * 100
        else:
            reduction_percent = 0.0

        compression_info ={
            'algorithm': self.config.get('algorithm', 'Indefinido'),
            'round': round_num,
            'client_id': self.client_id,
            'original_size_bytes': bytes_original,
            'delta_size_bytes': bytes_delta,
            'reduction_percent': reduction_percent
        }

        log_message = (
            f"Client {self.client_id} | "
            f"Algorithm: {self.config.get('algorithm')} | "
            f"Round: {round_num} | "
            f"Original Size: {bytes_original} bytes | "
            f"Delta RLE Size: {bytes_delta} bytes | "
            f"Reduction: {reduction_percent:.2f}%\n"
        )

        csv_filename = "client_compression_track.csv"
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csv_file:
            fieldnames = compression_info.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(compression_info)
        
        with open("compression_comparison_log.txt", "a") as log_file:
            log_file.write(log_message)