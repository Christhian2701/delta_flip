"""
FedBuff Aggregator.
"""
import numpy as np
from .base import BaseAggregator

class FedBuffAggregator(BaseAggregator):
    def __init__(self, config):
        self.config = config

    def aggregate(self, server, client_updates, round_num=None):
        """
        FedBuff Aggregation (Buffered Asynchronous SG)
        
        For simulation purposes (synchronous loop):
        - Treating the 'client_updates' as the buffer of K updates.
        - Assuming no staleness for this baseline (all trained on current model).
        
        Update Rule:
        w_new = w_old + eta_g * (1/K) * sum(w_i - w_old)
        
        This is equivalent to:
        delta = (1/K) * sum(w_i - w_old)
        w_new = w_old + eta_g * delta
        """
        K = len(client_updates)
        if K == 0:
            return server.global_weights
            
        server_lr = self.config.get('server_learning_rate', 1.0)
        
        # 1. Compute Average Delta
        # Initialize delta sum
        delta_sum = [np.zeros_like(w) for w in server.global_weights]
        
        for client_id, update in client_updates.items():
            client_weights = update['weights']
            
            # Compute w_i - w_old
            for i in range(len(client_weights)):
                delta_sum[i] += (client_weights[i] - server.global_weights[i])
                
        # 2. Compute Average Update
        avg_delta = [d / K for d in delta_sum]
        
        # 3. Apply Update
        new_weights = [server.global_weights[i] + server_lr * avg_delta[i] for i in range(len(server.global_weights))]
        
        return new_weights
