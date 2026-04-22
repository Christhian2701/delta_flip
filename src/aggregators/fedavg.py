"""
FedAvg Aggregator.
"""
import numpy as np
from .base import BaseAggregator

#imports for delta coding
from .delta_decompress import decode_rle, dequantize, unflatten, apply_deltas

class FedAvgAggregator(BaseAggregator):
    def aggregate(self, server, client_updates, round_num=None):

        #print("RUNNING FEDAVG AGGREGATOR")


        """
        Standard FedAvg aggregation (McMahan et al.)
        w = sum(n_k * w_k) / sum(n_k)
        """
        # Collect weights and sample counts
        new_weights = []
        total_examples = 0

        #adding deltacoding support in parallel (to compare accuracy)
        delta_weights = []


        for client_id, update in client_updates.items():
            num_samples = update['num_samples']
            weights = update['weights']

            #print(f'\n\tFEDAVG: Tipo de delta_weights para cliente {client_id}: {type(update.get("delta_weights", None))}')

            client_deltas = update.get('delta_weights', None)

            
            # Decompress if needed (simplified for Phase 1/2)
            # In simulation we pass weights directly          
            
            new_weights.append((num_samples, weights))

            if client_deltas is not None:

                #print("Delta Weights WAS FOUND IN CLIENT UPDATE, PROCEEDING TO add")
                
                delta_weights.append((num_samples, client_deltas))
            else:
                print("NO DELTAS DICTIONARY FOUND IN CLIENT UPDATE, USING FULL WEIGHTS INSTEAD")
                # If there are no deltas, uses general weights for consistency
                delta_weights.append((num_samples, weights))  
            total_examples += num_samples
            
        # Weighted Average
        # Initialize with zeros
        avg_weights = [np.zeros_like(w) for w in server.global_weights]
        dt_weights = [np.zeros_like(w) for w in server.global_weights]
        
        for num_samples, weights in new_weights:
            scale = num_samples / total_examples
            for i, w in enumerate(weights):
                avg_weights[i] += w * scale

        for num_samples, weights in delta_weights:
            scale = num_samples / total_examples
            for i, w in enumerate(weights):
                dt_weights[i] += w * scale
                
        return avg_weights, dt_weights
