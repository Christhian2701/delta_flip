"""
FedAvg Aggregator.
"""
import numpy as np
from .base import BaseAggregator

class FedAvgAggregator(BaseAggregator):
    def aggregate(self, server, client_updates, round_num=None):
        """
        Standard FedAvg aggregation (McMahan et al.)
        w = sum(n_k * w_k) / sum(n_k)
        """
        # Collect weights and sample counts
        new_weights = []
        total_examples = 0
        
        for client_id, update in client_updates.items():
            num_samples = update['num_samples']
            weights = update['weights']
            
            # Decompress if needed (simplified for Phase 1/2)
            # In simulation we pass weights directly
            
            new_weights.append((num_samples, weights))
            total_examples += num_samples
            
        # Weighted Average
        # Initialize with zeros
        avg_weights = [np.zeros_like(w) for w in server.global_weights]
        
        for num_samples, weights in new_weights:
            scale = num_samples / total_examples
            for i, w in enumerate(weights):
                avg_weights[i] += w * scale
                
        return avg_weights
