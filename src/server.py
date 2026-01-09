"""
FLIPS Server implementation.
"""
import numpy as np
import tensorflow as tf
from aggregators.fedavg import FedAvgAggregator
from aggregators.fedbuff import FedBuffAggregator
from aggregators.flips_aggregator import FLIPSAggregator
from aggregators.fedprox import FedProxAggregator
from aggregators.fedlama import FedLamaAggregator

class FLIPSServer:
    """
    FLIPS Server - handles model aggregation and global model management.

    Phase 1: FedAvg aggregation
    Phase 2: Will add importance-weighted aggregation
    Phase 3: Will add client selection based on mobility/network
    """

    def __init__(self, model, config):
        """
        Initialize FLIPS server.

        Args:
            model: Global Keras model
            config: Configuration dict
        """
        self.global_model = model
        self.config = config
        self.num_rounds = config['num_rounds']
        self.clients_per_round = config['clients_per_round']

        # Metrics tracking
        self.round_metrics = []
        self.global_weights = model.get_weights()
        
        # Initialize Aggregator Strategy
        self.aggregator = self._get_aggregator()

    def _get_aggregator(self):
        algo = self.config.get('algorithm', 'flips')
        if algo == 'fedavg':
            return FedAvgAggregator()
        elif algo == 'fedbuff':
            return FedBuffAggregator(self.config)
        elif algo == 'fedprox':
            return FedProxAggregator()
        elif algo == 'fedlama':
            return FedLamaAggregator(self.config)
        else:
            return FLIPSAggregator(self.config)

    def select_clients(self, clients):
        """
        Select clients for this round.
        Phase 1: Random selection
        Phase 3: Will add intelligent selection based on RSSI, mobility, etc.
        """
        # Random selection for now
        indices = np.random.choice(
            len(clients),
            size=min(self.clients_per_round, len(clients)),
            replace=False
        )
        return [clients[i] for i in indices]

    def run_round(self, round_num, clients, test_data, selected_clients=None):
        """
        Run one federated learning round.
        Phase 2: Adds weighted aggregation
        Phase 3: Accepts externally selected clients (mobility-aware)
        """
        # Select clients
        if selected_clients is None:
            selected_clients = self.select_clients(clients)
            
        selected_ids = [c.client_id for c in selected_clients]
        
        # FedLama: Determine active layers
        active_indices = None
        if isinstance(self.aggregator, FedLamaAggregator):
            active_indices = self.aggregator.get_active_indices(round_num, self.global_model)

        # Get current global weights
        self.global_weights = self.global_model.get_weights()

        # Client updates
        client_updates = {} # Changed to dictionary for easier access by client_id
        total_compressed_size = 0
        
        for client in selected_clients:
            # Phase 2 & 3: Get contact time from client propery
            # In Phase 3 this comes from vehicle mobility and is set before calling run_round
            contact_time = getattr(client, 'contact_time', 1.0)
            
            local_weights, num_samples, importance, size = client.train_local(
                self.global_weights, 
                active_indices=active_indices
            )
            
            # If size is 0 (uncompressed fedavg), estimate raw size
            if size == 0:
                # Estimate float32 size
                # Handle None in local_weights for FedLama
                size = sum([w.size * 4 for w in local_weights if w is not None])
            
            client_updates[client.client_id] = {
                'weights': local_weights,
                'num_samples': num_samples,
                'importance': importance,
                'contact_time': contact_time
            }
            total_compressed_size += size

        # 3. Aggregate using Strategy Pattern
        self.global_weights = self.aggregator.aggregate(self, client_updates, round_num=round_num)
        
        # FedLama: Update intervals
        if isinstance(self.aggregator, FedLamaAggregator) and active_indices:
             self.aggregator.update_intervals(active_indices, round_num)

        # Update global model
        self.global_model.set_weights(self.global_weights)

        # Evaluate
        X_test, y_test = test_data
        test_loss, test_accuracy = self.global_model.evaluate(
            X_test, y_test, verbose=0
        )

        # Collect metrics
        metrics = {
            'round': round_num,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'num_selected_clients': len(selected_clients),
            'avg_compression_bytes': total_compressed_size / len(selected_clients),
            'avg_local_accuracy': float(np.mean([c.local_accuracy for c in selected_clients])),
        }

        self.round_metrics.append(metrics)

        return metrics
