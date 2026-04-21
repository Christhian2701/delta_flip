"""
FLIPS Aggregator.
"""
import numpy as np

from aggregators.delta_decompress import apply_deltas, decode_rle, dequantize, unflatten
from .base import BaseAggregator

class FLIPSAggregator(BaseAggregator):
    def __init__(self, config):
        self.config = config

    def aggregate(self, server, client_updates, round_num=None):

        print("RUNNING BASE AGGREGATOR")
        #print(f'Client updates keys: {list(client_updates.keys())}')

        """
        Phase 2: Importance-Weighted Aggregation (Eq. 13)
        w^(t+1) = sum(I * N * tau * w) / sum(I * N * tau)
        """
        # Unwrap updates
        # update format: dict {client_id: {'weights': ..., 'num_samples': ..., ...}}
        
        # Initialize sums
        # Get first client updates to determine shape
        first_client_id = list(client_updates.keys())[0]
        first_client_weights = client_updates[first_client_id]['weights']
        
        # SIMPLIFIED IMPLEMENTATION FOR MVP:
        # We will iterate through model layers to map weights to importance
        
        #current_weight_idx = 0
        
        # Create zero-filled accumulators
        new_weights = [np.zeros_like(w) for w in server.global_weights]
        new_delta_weights = [np.zeros_like(w) for w in server.global_weights]

        new_weights = self.process_layer(server, client_updates, new_weights, weights_key='weights')

        new_delta_weights = self.process_layer(server, client_updates, new_delta_weights, weights_key='delta_weights')

        return new_weights, new_delta_weights


        '''if client_updates[list(client_updates.keys())[0]]['deltas_dictionary'] is not None:
            print("DELTAS DICTIONARY WAS FOUND IN BASE AGGREGATOR")

            for client_id, update in client_updates.items():


                #print(f"Keys in deltas_dictionary id {client_id}: {client_updates[client_id]['deltas_dictionary'].keys()}")

                deltas_dictionary = client_updates[client_id]['deltas_dictionary']

                decoded = decode_rle(deltas_dictionary['vector'])
                scale = deltas_dictionary['scale']
                dequantized = dequantize(decoded, scale)
                flat = unflatten(dequantized, deltas_dictionary['metadata'])
                rebuilt_weights = apply_deltas(server.global_weights, flat)

                client_updates[client_id]['rebuilt_weights'] = rebuilt_weights
                #ideia para depois, encontrar qual etapa ocorre IMEDIATAMENTE ANTES da agregação
                # para tratar os deltas lá e já chegar aqui sem precisar desse processamento
                

            #new_delta_weights = self.process_layer ???
            new_delta_weights = self.process_layer(server, client_updates, new_delta_weights, weights_key='rebuilt_weights')            

            return new_weights, new_delta_weights
'''



    def process_layer(self,server, client_updates, new_weights, weights_key='weights'):

        current_weight_idx = 0

        for layer in server.global_model.layers:
                if not layer.weights:
                    continue
                    
                # Get importance for this layer
                # If layer has no importance score, default to average or 1.0
                # For biases, usually same importance as kernel
                
                # Collect importance from all clients for this layer
                # But wait, Eq 13 says: sum(I_v * ... * w_v) / sum(I_v * ...)
                # So we iterate clients
                
                num_layer_weights = len(layer.get_weights())
                
                for w_local_idx in range(num_layer_weights):
                    w_global_idx = current_weight_idx + w_local_idx
                    
                    sum_weighted_w = np.zeros_like(new_weights[w_global_idx])
                    sum_factors = 0.0 + 1e-10 # Avoid diff by zero
                    
                    for client_id, update in client_updates.items():
                        weights = update[weights_key]
                        num_samples = update['num_samples']
                        client_importance = update['importance']
                        contact_time = update['contact_time']
                        
                        # Get importance for this layer from this client
                        imp = client_importance.get(layer.name, 1.0)
                        
                        # Factor = I * N * tau
                        factor = imp * num_samples * contact_time
                        
                        sum_weighted_w += weights[w_global_idx] * factor
                        sum_factors += factor
                        
                    new_weights[w_global_idx] = sum_weighted_w / sum_factors
                    
                current_weight_idx += num_layer_weights

        '''print("NEW WEIGHTS AFTER PROCESSING LAYER:")
        for idx, w in enumerate(new_weights):
            print(f"Layer {idx} weights shape: {w.shape}") '''
        return new_weights
