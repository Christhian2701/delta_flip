"""
alteração de modelo
FLIPS Client implementation.

MODIFIDAÇÕES FEITAS AQUI
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import gzip
import pickle
import sys
import copy
import math
import os
import csv

class FLIPSClient:
    """
    FLIPS Client - handles local training and model updates.

    Phase 1: Basic local training
    Phase 2: Will add SHAP, pruning, quantization
    Phase 3: Will add vehicle properties (position, velocity)
    """

    def __init__(self, client_id, data, model, config):
        """
        Initialize FLIPS client.

        Args:
            client_id: Unique client identifier
            data: Dict with 'X_train', 'y_train', 'X_val', 'y_val'
            model: Keras model (will be cloned)
            config: Configuration dict
        """
        self.client_id = client_id
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.num_samples = len(self.y_train)

        #adding round tracking for compression analysis
        self.round_track = {
            'max_rounds' : config.get('num_rounds', 10),
            'algorithm': config.get('algorithm'),
            'current_round': 0
        }

        #print(f"Client {self.client_id} round {self.round_track['current_round']} of {self.round_track['max_rounds']}, initialized with {self.num_samples} samples.")

        # Clone model for this client
        self.old_model = None # para manter modelo antigo e usar em delta coding
        #self.model = keras.models.clone_model(model)
        self.model = model
        self.model.set_weights(model.get_weights())
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        

        # Configuration
        self.config = config
        self.local_epochs = config['local_epochs']
        self.batch_size = config['batch_size']

        # Metrics
        self.local_accuracy = 0.0
        self.local_loss = 0.0
        
        # Phase 3: Mobility context & Metrics
        self.contact_time = 1.0 
        self.rssi_norm = 1.0
        self.dropout_count = 0
        self.training_time = np.random.uniform(0.5, 2.0) # Simulated training time (s)
        self.device_density = 0.0 # Will be updated by server/simulation

    def get_context_factor(self):
        """
        Compute context factor omega_k (Eq. 305).
        omega_k = min(1, g1*RSSI + g2*(1/(1+dropout)) + g3*tau)
        """
        g1 = self.config.get('gamma1_rssi', 0.33)
        g2 = self.config.get('gamma2_dropout', 0.33)
        g3 = self.config.get('gamma3_contact', 0.33)
        
        term1 = g1 * self.rssi_norm
        term2 = g2 * (1.0 / (1.0 + self.dropout_count))
        term3 = g3 * self.contact_time
        
        return min(1.0, term1 + term2 + term3)
    
    def bn_train_fedprox(self, global_weights):
        """
        FedProx Local Training: L(w) + (mu/2) * ||w - w^t||^2
        """
        mu = self.config.get('mu', 0.01)
        optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'])
        
        # FIX: Snapshot the global trainable weights directly from the model.
        # Since self.model.set_weights(global_weights) was called right before this,
        # self.model.trainable_variables currently holds the exact global trainable weights.
        # We use tf.identity to create a detached copy of these tensors for the loss calculation.
        global_trainable_weights = [tf.identity(w) for w in self.model.trainable_variables]

        # Prepare dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        for epoch in range(self.local_epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    
                    # Original Loss
                    loss_value = loss_fn(y_batch_train, logits)
                    
                    # Proximal Term
                    proximal_term = 0.0
                    
                    # Iterate and compute L2 loss using aligned trainable arrays
                    for i, w in enumerate(self.model.trainable_variables):
                        proximal_term += tf.nn.l2_loss(w - global_trainable_weights[i])
                        
                    loss_value += (mu / 2.0) * proximal_term
                    
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
        # Evaluate on validation to set local metrics
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return val_loss, val_acc

    def _train_fedprox(self, global_weights):
        """
        FedProx Local Training: L(w) + (mu/2) * ||w - w^t||^2
        """
        mu = self.config.get('mu', 0.01)
        optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'])
        
        # Convert global weights to tensors for comparison
        # Only consider trainable weights for the proximal term
        global_kernel_weights = [tf.convert_to_tensor(w) for w in global_weights]

        # Prepare dataset
        # Create dataset from numpy arrays for easier iteration
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        for epoch in range(self.local_epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    
                    # Original Loss
                    loss_value = loss_fn(y_batch_train, logits)
                    
                    # Proximal Term
                    proximal_term = 0.0
                    # Iterate through trainable variables of the model
                    # and corresponding global weights
                    for i, w in enumerate(self.model.trainable_variables):
                        # Ensure we only compare weights that exist in both
                        if i < len(global_kernel_weights):
                            proximal_term += tf.nn.l2_loss(w - global_kernel_weights[i])
                        
                    loss_value += (mu / 2.0) * proximal_term
                    
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
        # Evaluate on validation to set local metrics
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return val_loss, val_acc

    def train_local(self, global_weights, active_indices=None):
        """
        Perform local training and FLIPS specific operations.
        
        Args:
            global_weights: List of numpy arrays
            active_indices: Optional list of indices to return updates for (FedLama)
        """ 
        # os pesos do modelo global são salvos a parte pra usar delta coding
        self.old_model = [np.copy(w) for w in global_weights]

        # Set global weights
        self.model.set_weights(global_weights)

        # FedProx Implementation
        if self.config.get('algorithm') == 'fedprox':
            # FedProx: Default mu
            if 'mu' not in self.config:
                self.config['mu'] = 0.01 # Default mu if not provided
            self.local_loss, self.local_accuracy = self._train_fedprox(global_weights)
            
            # Return standard FedAvg-like update structure
            local_weights = self.model.get_weights()
            if active_indices is not None:
                local_weights = [w if i in active_indices else None for i, w in enumerate(local_weights)]

            
            #print("RETURN ON FEDPROX IMPLEMENTATION")

            try:
                _, deltas_dictionary = self.quantize_and_compress()
            except Exception as e:
                print(f"Error during quantization/compression for client {self.client_id}: {e}")
                deltas_dictionary = None

            return local_weights, self.num_samples, {}, 0, deltas_dictionary
            
        # Standard FedAvg / FLIPS training
        # Train locally
        # In simulation: training time is just a value, but we fit the model
        history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.local_epochs,
            verbose=0,
            validation_data=(self.X_val, self.y_val)
        )

        # Update metrics
        self.local_loss = history.history['loss'][-1]
        self.local_accuracy = history.history['val_accuracy'][-1]

        # FedAvg Baseline check
        algorithm = self.config.get('algorithm', 'flips')
        if algorithm in ['fedavg', 'fedlama']:
            local_weights = self.model.get_weights()
            
            # FedLama: Filter weights
            if active_indices is not None:
                local_weights = [w if i in active_indices else None for i, w in enumerate(local_weights)]
            
            # Return empty importance/metrics for consistent signature            
            #print("RETURN ON Standard FedAvg") 

            try:
                _, deltas_dictionary = self.quantize_and_compress()
            except Exception as e:
                print(f"Error during quantization/compression for client {self.client_id}: {e}")
                deltas_dictionary = None
            #else:
                #print(f"Quantization and compression successful for client {self.client_id}")
            

            return local_weights, self.num_samples, {}, 0, deltas_dictionary # 0 compressed size (raw)


        # Note on Fedbuff, the algorithm doesn't have an specific local training procedure, instead it runs just as flips (Phase 2 and Phase 3), will leave as such, but worth looking into for what was the intention in te article

        # FedBuff Baseline check (to be done)
        #algorithm = self.config.get('algorithm', 'fedbuff')

        # Phase 2: Compute Importance & Apply Context (Eq. 292)
        omega = self.get_context_factor()
        raw_importance = self.compute_shap_importance()
        
        # Apply omega to importance (Eq 292)
        importance_scores = {k: v * omega for k, v in raw_importance.items()}

        # Phase 2: Importance Normalization (Eq. 372)
        # I_k^l <- I_k^l / max(I_k)
        if importance_scores:
            max_imp = max(importance_scores.values())
            if max_imp > 0:
                importance_scores = {k: v / max_imp for k, v in importance_scores.items()}
        
        # Phase 2: Pruning (Eq 412, 418)
        # We prune using the normalized scores. Eq 418 uses scaling (I/Imax), 
        # so normalizing to [0,1] first is mathematically equivalent and cleaner.
        pruning_ratio = self.prune_model(importance_scores, self.contact_time)
        
        # Phase 2: Quantization & Compression
        compressed_size, deltas_dictionary = self.quantize_and_compress()

        # Return updated weights
        local_weights = self.model.get_weights()

        #print(f'INFO on delta_dictionary for client {self.client_id}:{list(deltas_dictionary.keys()) if deltas_dictionary else "No deltas"}')

        return local_weights, self.num_samples, importance_scores, compressed_size, deltas_dictionary

    def compute_shap_importance(self):
        """
        Compute SHAP-based layer importance using Deep SHAP.

        Implements Equations 236-241 from the paper:
        - φ_k^l(x_i): SHAP value for layer l on sample x_i
        - I_k^l = (1/|D_val|) × Σ |φ_k^l(x_i)|

        For computational efficiency, we use a gradient-based approximation
        that captures layer contribution to model predictions, similar to
        SHAP's concept of feature attribution.

        Returns:
            dict: Layer importance scores {layer_name: importance_value}
        """
        layer_importance = {}

        # Sample validation data (Eq 241 uses validation set)
        num_shap_samples = min(
            self.config.get('shap_samples', 100),
            len(self.X_val)
        )

        # If validation set is very small, use training data subset
        if len(self.X_val) < 10:
            sample_X = self.X_train[:num_shap_samples]
            sample_y = self.y_train[:num_shap_samples]
        else:
            sample_X = self.X_val[:num_shap_samples]
            sample_y = self.y_val[:num_shap_samples]

        # METHOD 1: Gradient-based layer importance (efficient SHAP approximation)
        # This captures how each layer's parameters affect the loss, similar to
        # SHAP's marginal contribution concept but computed via gradients

        try:
            with tf.GradientTape(persistent=True) as tape:
                # Forward pass
                predictions = self.model(sample_X, training=False)

                # Compute loss (Eq 147 in paper)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    sample_y, predictions
                )
                loss = tf.reduce_mean(loss)

            # Compute gradients and importance for each layer
            for layer in self.model.layers:
                if not layer.trainable_weights:
                    continue

                # Get kernel (main weights, not biases)
                kernel = layer.trainable_weights[0]

                # Compute gradient of loss w.r.t. layer weights
                grad = tape.gradient(loss, kernel)

                if grad is None:
                    continue

                # Layer importance = mean(|gradient * weight|)
                # This approximates SHAP's attribution:
                # - gradient: how much changing the weight affects loss
                # - weight: current magnitude of the parameter
                # - product: marginal contribution (similar to SHAP value)
                importance = tf.reduce_mean(tf.abs(grad * kernel))

                layer_importance[layer.name] = float(importance.numpy())

            del tape  # Clean up persistent tape

        except Exception as e:
            print(f"Warning: Gradient-based SHAP computation failed: {e}")
            print("Falling back to activation-based importance...")

            # METHOD 2: Activation-based importance (fallback)
            # Measure layer output variance as importance proxy
            layer_outputs = self._get_layer_outputs(sample_X)

            for layer_name, output in layer_outputs.items():
                # Importance = variance of activations
                # High variance = layer is differentiating between inputs
                if output is not None:
                    importance = float(np.var(output))
                    layer_importance[layer_name] = importance

        # If still no importance computed, use weight magnitude as last resort
        if not layer_importance:
            print("Warning: Using weight magnitude fallback for SHAP importance")
            for layer in self.model.layers:
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    w = layer.get_weights()[0]
                    layer_importance[layer.name] = float(np.mean(np.abs(w)))

        return layer_importance

    def _get_layer_outputs(self, input_data):
        """
        Helper: Get activations from each layer for activation-based importance.

        Args:
            input_data: Input samples

        Returns:
            dict: {layer_name: activation_array}
        """
        layer_outputs = {}

        # Create intermediate models to extract layer outputs
        for i, layer in enumerate(self.model.layers):
            if not layer.trainable_weights:
                continue

            try:
                # Create model up to this layer
                intermediate_model = keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )

                # Get activations
                activations = intermediate_model.predict(
                    input_data,
                    verbose=0,
                    batch_size=32
                )

                layer_outputs[layer.name] = activations

            except Exception as e:
                # Skip layers that can't be extracted
                continue

        return layer_outputs

    def prune_model(self, importance_scores, contact_time):
        """
        Phase 2: Prune model based on importance and contact time.
        Eq 412: theta_base(v) = theta_base * (1 + alpha * tau)
        Eq 418: theta_k^l = theta_base(v) * (1 - I_k^l / I_max)
        """
        theta_base_val = self.config.get('theta_base', 0.01)
        alpha = self.config.get('alpha_contact', 0.5) # Paper implies positive alpha
        
        # Eq 412: Dynamic base threshold
        # If the paper implies "prune aggressively" for SHORT contact, then alpha should likely be negative 
        # or the formula in paper has specific constraints. 
        # We will implement EXACTLY as Eq 412 written.
        adjusted_base = theta_base_val * (1.0 + alpha * contact_time)

        max_importance = max(importance_scores.values()) if importance_scores else 1.0
        
        total_params = 0
        pruned_params = 0
        
        for layer in self.model.layers:
            if layer.name not in importance_scores:
                continue
                
            importance = importance_scores[layer.name]
            
            # Eq 418: Layer-specific threshold
            threshold = adjusted_base * (1.0 - importance / max_importance)
            
            weights = layer.get_weights()
            if not weights: continue
            
            # Apply pruning mask
            w = weights[0]
            mask = np.abs(w) > threshold
            w_pruned = w * mask
            
            pruned_count = w.size - np.sum(mask)
            total_params += w.size
            pruned_params += pruned_count
            
            # Update weights
            weights[0] = w_pruned
            layer.set_weights(weights)
            
        return pruned_params / max(1, total_params)

    
    # onde melhor colocar o delta coding
    # tem que manter o modelo antigo para manter o delta

    
    
    def quantize_and_compress(self):
        """
        Phase 2: Quantize weights (float32 -> int8) + Gzip.
        Returns serialized size in bytes.
        """
        weights = self.model.get_weights()
        # adição para delta coding
        old_weights = self.old_model if self.old_model is not None else [np.zeros_like(w) for w in weights]

        deltas = self.get_deltas(old_weights, weights)

        deltas_dictionary = self.get_flat(deltas)

        deltas_dictionary['vector'], deltas_dictionary['scale'] = self.uniform_quantization(deltas_dictionary['vector'])

        encoded_deltas = self.rle_encoding(deltas_dictionary['vector'])

        deltas_dictionary['vector'] = encoded_deltas

            # preparo para comparação de compressão 

        delta_serialized = pickle.dumps(deltas_dictionary)
        delta_compressed = gzip.compress(delta_serialized)
        delta_size_bytes = len(delta_compressed)

        # fim delta coding
        
        # Quantization Simulation (casting)
        # In real deployment: tf.quantization.quantize
        quantized_weights = []
        for w in weights:
            # Scale to range partitions, etc. 
            # Simple simulation: cast to float16 then compress
            quantized_weights.append(w.astype(np.float16))
            
        # Serialize
        serialized = pickle.dumps(quantized_weights)
        compressed = gzip.compress(serialized)
        original_size_bytes = len(compressed)

        self.comparison(original_size_bytes, delta_size_bytes)
        
        return len(compressed), deltas_dictionary

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy

    # métodos para delta coding

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
        """Uniform quantization of deltas."""

        max_abs  =  np.max(np.abs(vector))
        scale = max_abs / 127.0
        
        result = np.clip(np.round(vector / scale), -127, 127).astype(np.int8)

        return result, scale

    def rle_encoding(self, values):

        sentinel =  -128
        max_run = 127 # máximo de zeros seguidos para 8 bits

        encoded = []

        i, n = 0, len(values) # contadores para iteração

        while i < n:
            if abs(values[i]) == 0:
                sequence_length = 0
                while i < n and abs(values[i]) == 0:
                    sequence_length += 1
                    i += 1
                
                # quebra a sequência pra caber no tipo de vetor (int8)
                while sequence_length > 0:
                    chunk = min(sequence_length, max_run)
                    encoded.append(sentinel)
                    encoded.append(chunk)
                    sequence_length -= chunk
            else:
                encoded.append(values[i])
                i += 1

        return encoded

    def comparison(self, bytes_original, bytes_delta):

        if bytes_original > 0:
            reduction_percent = (1 - (bytes_delta / bytes_original)) * 100
        else:
            reduction_percent = 0.0

        current_round = self.round_track.get('current_round', 0)

        compression_info ={
            'algorithm': self.config.get('algorithm', 'Indefinido'),
            'round': current_round,
            'client_id': self.client_id,
            'original_size_bytes': bytes_original,
            'delta_size_bytes': bytes_delta,
            'reduction_percent': reduction_percent
        }

        #originalmente usava só log_message
        log_message = (
            f"Client {self.client_id} | "
            f"Algorithm: {self.config.get('algorithm')} | "
            f"Round: {self.config.get('current_round', 0)} | "
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
        
        # Print to console so you can see it running
        #print(log_message.strip())

        # Append to a text file for later analysis
        # Using 'a' mode to append, so it records every client in every round
        with open("compression_comparison_log.txt", "a") as log_file:
            log_file.write(log_message)
        
        self.round_track['current_round'] += 1
    
