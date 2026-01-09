"""
FedProx Aggregator.
"""
from .fedavg import FedAvgAggregator

class FedProxAggregator(FedAvgAggregator):
    """
    FedProx uses the same aggregation logic as FedAvg (weighted average).
    The difference is only in the local client training (proximal term).
    """
    pass
