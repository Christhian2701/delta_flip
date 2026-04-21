================================================================================
ANALYSIS: FLIPS Federated Learning - Model Aggregation & Delta Coding Integration
================================================================================

================================================================================
1. WHICH PART TESTS AND AGGREGATES MODELS FROM CLIENTS?
================================================================================

The model aggregation happens in TWO key places:

A) SERVER ORCHESTRATION (src/server.py:68-143)
   └─> Method: FLIPSServer.run_round()
   
   Flow:
   1. Select clients (line 76)
   2. Get global weights (line 86)
   3. Loop through selected_clients, call client.train_local() (lines 92-114)
   4. Collect client_updates dictionary with:
      - weights: local model weights
      - num_samples: dataset size
      - importance: SHAP scores (for FLIPS)
      - contact_time: mobility factor
   5. Call aggregator.aggregate() (line 117)
   6. Evaluate on test_data (lines 127-130)
   7. Store metrics (lines 133-142)

B) AGGREGATOR IMPLEMENTATIONS (src/aggregators/*.py)
   └─> Base Interface: src/aggregators/base.py
   └─> Concrete Implementations:
   
   * FLIPSAggregator (flips_aggregator.py:11-71)
     - Equation 13: w^(t+1) = sum(I * N * tau * w) / sum(I * N * tau)
     - Weights by: importance × dataset_size × contact_time
   
   * FedAvgAggregator (fedavg.py:8-36)
     - Equation: w = sum(n_k * w_k) / sum(n_k)
     - Weights by: dataset_size only (baseline, no compression)
   
   * FedBuffAggregator, FedProxAggregator, FedLamaAggregator
     - Variants with buffered SGD, proximal term, adaptive intervals

================================================================================
2. HOW DOES THIS PART CONNECT TO OTHERS?
================================================================================

CALL CHAIN:

run_experiment.py (entry point)
    │
    ├─> Creates CNN model (build_cnn())
    ├─> Loads & partitions CIFAR-100 data (load_cifar100_noniid())
    ├─> Creates FLIPSClient instances for each partition
    ├─> Creates FLIPSServer with aggregator
    └─> Calls run_federated_learning() ──────────────────────────────┐
                                                                    │
simulation.py:run_federated_learning()                               │
    │                                                               │
    ├─> Creates Vehicle/BaseStation (mobility.py)                   │
    ├─> Creates ClientSelector (selection.py)                        │
    │                                                               │
    └─> LOOP: for each round:                                        │
            ├─> Update vehicle positions                             │
            ├─> Estimate contact_time for each client                │
            ├─> selector.select_clients() → filter by RSSI           │
            └─> server.run_round() ◄────────────────────────────────┘
                            │
                            ▼
server.py:FLIPSServer.run_round()
    │
    ├─> Select clients (or use pre-selected)
    ├─> Loop: client.train_local(global_weights)
    │       │
    │       ▼
    │   client.py:FLIPSClient.train_local()
    │       │
    │       ├─> Save old_model (delta coding prep, line 137)
    │       ├─> Set global weights
    │       ├─> Train locally (model.fit())
    │       ├─> Compute SHAP importance
    │       ├─> Apply pruning
    │       ├─> quantize_and_compress() ← DELTA CODING HERE
    │       └─> Return: (weights, num_samples, importance, size)
    │
    └─> aggregator.aggregate(server, client_updates)
            │
            ▼
        AGGREGATORS (flips_aggregator.py, fedavg.py, etc.)
            │
            └─> Return new global_weights

DATA FLOW SUMMARY:
┌──────────────┐    global_weights     ┌──────────────┐
│    Server    │ ◄─────────────────────│   Aggregator  │
│  (server.py) │                       │ (flips_agg)  │
└──────┬───────┘                       └──────────────┘
       │
       │ train_local()
       ▼
┌──────────────┐    compressed_size    ┌──────────────┐
│    Client    │ ─────────────────────►│   Simulation │
│ (client.py)  │                       │(simulation.py)│
└──────────────┘                       └──────────────┘

KEY CONNECTIONS:
- client.py:137-206: Stores old weights, trains, computes importance, prunes
- client.py:396-438: quantize_and_compress() - applies delta + RLE encoding
- server.py:97-114: Receives compressed_size, stores in client_updates
- server.py:138: Reports avg_compression_bytes in metrics

================================================================================
3. HOW TO ADD PARALLEL COMPARISON WITH DELTA CODING + QUANTIZATION
================================================================================

GOAL: Run experiments comparing accuracy of models WITH and WITHOUT
      delta coding + quantization compression, in parallel.

CURRENT STATE:
- client.py:396-438 already implements delta coding + quantization
- However, the CURRENT implementation:
  1. Compresses weights for transmission simulation
  2. BUT still sends FULL weights to aggregator (line 206-208)
  3. Aggregation happens on uncompressed weights

WHAT NEEDS TO CHANGE:

═══════════════════════════════════════════════════════════════════════════════
STEP 1: Modify client.py to optionally return COMPRESSED model updates
═══════════════════════════════════════════════════════════════════════════════

Location: src/client.py:128-208 (train_local method)

Add a flag to control compression behavior:
```python
def train_local(self, global_weights, active_indices=None, use_compression=True):
    # ... existing code ...
    
    if use_compression:
        # Return deltas instead of full weights
        deltas = self.get_deltas(self.old_model, self.model.get_weights())
        return deltas, self.num_samples, importance_scores, compressed_size
    else:
        # Return full weights (baseline)
        return self.model.get_weights(), self.num_samples, {}, 0
```

═══════════════════════════════════════════════════════════════════════════════
STEP 2: Add decompression method to client or create new DecompressedAggregator
═══════════════════════════════════════════════════════════════════════════════

Location: src/aggregators/ (new file: compressed_aggregator.py)

The aggregator needs to reconstruct full weights from compressed deltas:
```python
class CompressedFLIPSAggregator(BaseAggregator):
    def aggregate(self, server, client_updates, round_num=None):
        # Decompress deltas back to weights
        decompressed_updates = {}
        
        for client_id, update in client_updates.items():
            deltas = update['weights']  # These are compressed deltas
            old_weights = self.get_previous_weights(client_id)  # Need to track
            weights = self.decompress(deltas, old_weights)
            decompressed_updates[client_id] = {
                'weights': weights,
                'num_samples': update['num_samples'],
                'importance': update['importance'],
                'contact_time': update['contact_time']
            }
        
        # Now aggregate as normal (use FLIPS or FedAvg logic)
        return self._aggregate_standard(server, decompressed_updates)
```

═══════════════════════════════════════════════════════════════════════════════
STEP 3: Modify server.py to support dual-track aggregation
═══════════════════════════════════════════════════════════════════════════════

Location: src/server.py:68-143 (run_round method)

Option A - Run two parallel experiments (simpler):
```python
# Create two servers: one with compressed, one without
server_compressed = FLIPSServer(model, config, use_compression=True)
server_uncompressed = FLIPSServer(model, config, use_compression=False)

# In each round, run both:
metrics_compressed = server_compressed.run_round(...)
metrics_uncompressed = server_uncompressed.run_round(...)

# Store both for comparison
```

Option B - Add algorithm variants (cleaner):
```python
# Add new algorithm types
algorithms = ['flips', 'flips_compressed', 'fedavg', 'fedavg_compressed', ...]

# In server._get_aggregator():
if algo == 'flips_compressed':
    return CompressedFLIPSAggregator(self.config, use_delta=True)
```

═══════════════════════════════════════════════════════════════════════════════
STEP 4: Modify run_experiment.py to run comparative experiments
═══════════════════════════════════════════════════════════════════════════════

Location: run_experiment.py:108-142

Change the algorithm list to include both variants:
```python
algorithms = [
    'flips',           # Original (no delta coding)
    'flips_compressed', # With delta coding + quantization
    'fedavg',
    'fedavg_compressed',
    # ... etc
]

for algo in algorithms:
    # ... existing setup code ...
    
    # Run experiment
    server = run_federated_learning(clients, server, test_data, run_config)
    results[algo] = server.round_metrics
```

═══════════════════════════════════════════════════════════════════════════════
STEP 5: Track and log comparison metrics
═══════════════════════════════════════════════════════════════════════════════

Add to server.py metrics (or create new comparison file):
```python
metrics = {
    'round': round_num,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'compression_ratio': ...,       # NEW: bytes_saved / bytes_original
    'accuracy_vs_baseline': ...,     # NEW: delta from uncompressed
}
```

Save to new_results/metrics_comparison.json

================================================================================
DETAILED IMPLEMENTATION STEPS
================================================================================

1. CREATE src/aggregators/compressed_aggregator.py
   - Decompress deltas to weights
   - Apply standard FLIPS/FedAvg aggregation
   - Track previous global weights per client

2. MODIFY src/client.py
   - Add use_compression parameter to train_local()
   - Ensure quantize_and_compress() returns proper delta dictionary

3. MODIFY src/server.py
   - Add use_compression flag to __init__()
   - Pass flag to clients
   - Store compression metrics

4. MODIFY run_experiment.py
   - Add _compressed algorithm variants to comparison
   - Track and save comparative metrics

5. CREATE comparison visualization
   - Plot accuracy curves: baseline vs compressed
   - Show compression ratio over rounds
   - Calculate accuracy degradation %

================================================================================
FILE CHANGES SUMMARY
================================================================================

MODIFIED:
- src/client.py: Add use_compression parameter, return deltas when enabled
- src/server.py: Add compression metrics, dual-track support
- run_experiment.py: Add compressed algorithm variants

NEW:
- src/aggregators/compressed_aggregator.py: Decompress + aggregate
- new_results/comparison_metrics.json: Parallel comparison results

================================================================================
TESTING APPROACH
================================================================================

After implementation, run:
  python run_experiment.py --config configs/config.yaml

Expected output in new_results/comparison_metrics.json:
{
  "flips": {...},           # Original accuracy curve
  "flips_compressed": {...}, # With delta coding - should be similar
  "fedavg": {...},
  "fedavg_compressed": {...}
}

Key metrics to compare:
1. Final test accuracy (should be within ~1-2% of uncompressed)
2. Compression ratio (bytes_saved / bytes_original)
3. Convergence speed (rounds to reach 90% of final accuracy)
4. Training time (should be similar)

================================================================================
