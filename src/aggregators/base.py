"""
Base Aggregator Interface.
"""

class BaseAggregator:
    def aggregate(self, server, client_updates, round_num=None):
        """
        Aggregate client updates.
        
        Args:
            server: The FLIPSServer instance (for access to global weights/model if needed)
            client_updates: Dictionary of client updates 
                          {client_id: {'weights': ..., 'num_samples': ..., ...}}
                          
        Returns:
            list: New global weights
        """
        raise NotImplementedError
