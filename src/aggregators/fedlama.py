"""
FedLama Aggregator.
"""
import numpy as np
from .base import BaseAggregator

class FedLamaAggregator(BaseAggregator):
    """
    FedLAMA: Federated Learning with Adaptive Model Aggregation.
    Adapts aggregation interval per layer based on discrepancy.
    """
    def __init__(self, config):
        self.config = config
        self.layer_intervals = {} # {layer_index: interval}
        self.layer_last_update = {} # {layer_index: round_num}
        self.min_interval = 1
        self.max_interval = config.get('fedlama_max_interval', 5)
        self.discrepancy_threshold = config.get('fedlama_threshold', 0.01)

    def get_active_indices(self, round_num, model):
        """
        Determine which weight indices are active for this round.
        Returns list of indices (into model.get_weights() list).
        """
        active_indices = []
        weights = model.get_weights()
        
        # Initialize intervals if needed
        if not self.layer_intervals:
            for i in range(len(weights)):
                self.layer_intervals[i] = 1
                self.layer_last_update[i] = 0
                
        for i in range(len(weights)):
            interval = self.layer_intervals[i]
            last = self.layer_last_update[i]
            
            # Aggregate if interval passed or first round
            if (round_num - last) >= interval or round_num == 0:
                active_indices.append(i)
                
        return active_indices

    def aggregate(self, server, client_updates, round_num=None):
        """
        Aggregate only active layers.
        Updates intervals based on discrepancy.
        """
        # Determine active indices from the updates (assuming all clients sent same partial)
        # We need to know which indices correspond to what exists in client_updates
        # But wait, client_updates for FedLama will contain sparse lists (None for skipped)
        
        # We need the server's current round to know what *should* have been sent
        # Accessing private server state is a bit messy but allowed in this tightly coupled design
        # Better: get it from context or pass it in. 
        # We'll assume the client_updates structure: {client_id: {weights: [w0, None, w2...]}}
        
        first_client = list(client_updates.values())[0]
        received_weights = first_client['weights']
        
        active_indices = [i for i, w in enumerate(received_weights) if w is not None]
        
        # Standard FedAvg on ACTIVE weights
        new_global_weights = list(server.global_weights) # Copy
        
        total_samples = sum(u['num_samples'] for u in client_updates.values())
        
        for idx in active_indices:
            weighted_sum = np.zeros_like(new_global_weights[idx])
            
            # Discrepancy calculation: Variance of updates?
            # Or L2 distance from global?
            # Paper: Discrepancy delta_l = || w_l - w_global || / || w_global ||
            # Or variance across clients? 
            # "Discrepancy of model parameters in different layers"
            # Simple heuristic: If updates are very different from each other, aggregate OFTEN (low interval).
            # If updates are similar (low variance), aggregate LESS OFTEN.
            
            updates_stack = []
            
            for client_id, update in client_updates.items():
                w_client = update['weights'][idx]
                n_client = update['num_samples']
                
                weighted_sum += w_client * n_client
                updates_stack.append(w_client)
                
            # Update global
            new_global_weights[idx] = weighted_sum / total_samples
            
            # Adopt Discrepancy-based Interval Update
            # Metric: Normalized Variance of client updates for this layer
            updates_stack = np.array(updates_stack)
            if len(updates_stack) > 1:
                variance = np.var(updates_stack, axis=0)
                norm_variance = np.mean(variance) # Scalar proxy
                
                # Update Interval
                current_interval = self.layer_intervals[idx]
                
                if norm_variance < self.discrepancy_threshold:
                    # Low discrepancy -> Can relax (increase interval)
                    self.layer_intervals[idx] = min(self.max_interval, current_interval + 1)
                else:
                    # High discrepancy -> Needs frequent updates (decrease interval)
                    self.layer_intervals[idx] = max(self.min_interval, current_interval - 1)
            
            # Mark update
            # We don't have easy access to current round num inside Aggregate without passing it
            # But we can assume this is the update for the *current* round of aggregation
            # Let's verify if we need exact round number. 
            # Yes, to check "if (round - last) >= interval".
            # We updated 'last' in get_active_indices? No, that's a query.
            # We should update 'last_update' HERE.
            # But we don't know 'round_num'.
            # Modification: Pass round_num to aggregate?
            # Or just increment a counter in aggregator.
            
        # Update last_updates for active indices
        # We will track internal 'current_round' or similar?
        # Let's rely on server passing round if possible, or hack it:
        # We will assume get_active_indices was called just before.
        # But safer: Modify BaseAggregator.aggregate signature or Server passes round.
        
        return new_global_weights

    def update_intervals(self, active_indices, round_num):
         for idx in active_indices:
             self.layer_last_update[idx] = round_num
