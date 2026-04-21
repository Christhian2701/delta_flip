# Delta Aggregation Analysis & Future Steps

## Current Project State

### Aggregators & Delta Support Status

| Aggregator | File | Delta Support | Returns Tuple |
|-----------|------|--------------|--------------|
| FedAvg | `fedavg.py` | ✓ Implemented | `avg_weights, dt_weights` |
| FLIPS | `flips_aggregator.py` | Partial (not fully working) | Not returning tuple |
| FedProx | `fedprox.py` | ✗ None | Single value |
| FedBuff | `fedbuff.py` | ✗ None | Single value |
| FedLama | `fedlama.py` | ✗ None | Single value |

---

## How Delta Works in Current Project

### Flow: Client → Server → Aggregator

```
client.train_local()
  │
  ├─ Save global_weights to self.old_model (for delta)
  ├─ Train model locally
  ├─ If algorithm == 'flips':
  │   └─ Compute SHAP importance
  │   └─ Prune model
  │   └─ quantize_and_compress() → returns deltas_dictionary
  │
  └─ Return: (weights, num_samples, importance, size, deltas_dictionary)

server.run_round()
  │
  └─ client_updates[id] = {
        'weights': ...,
        'num_samples': ...,
        'importance': ...,
        'contact_time': ...,
        'deltas_dictionary': deltas_dictionary  ← Only for 'flips'!
    }

aggregator.aggregate()
  │
  ├─ If has deltas:
  │   ├─ decode_rle(vector)
  │   ├─ dequantize(decoded, scale)
  │   ├─ unflatten(dequantized, metadata)
  │   └─ apply_deltas(global_weights, flat)
  │
  └─ Return: (standard_weights, delta_weights)
```

---

## Current Issues

### 1. Only 'flips' Algorithm Produces Deltas

**client.py lines 173-184** - fedavg/fedlama return early WITHOUT deltas:
```python
if algorithm in ['fedavg', 'fedlama']:
    return local_weights, self.num_samples, {}, 0, None  # None = no deltas!
```

**Fix needed**: Call `quantize_and_compress()` before returning for fedavg/fedlama

### 2. FLIPSAggregator Not Fully Working

Current `flips_aggregator.py`:
- Decodes deltas into local variable (overwrites each client)
- Never accumulates properly
- Doesn't return tuple

**Fix needed**: Option B implementation (weight_key parameter)

### 3. Metrics Logging

Current server.py logs both accuracies:
```python
'test_accuracy': standard,      # From global_model
'delta_accuracy': from_delta   # From delta_model
```

---

## Plan for Full Comparison

### Phase 1: Fix FedAvg (ALREADY WORKING ✓)
- fedavg.py returns tuple correctly
- server.py evaluates both models
- Results saved to metrics

### Phase 2: Fix FLIPS Aggregator
Implement Option B:
1. Modify `process_layer(weight_key='weights')`
2. Decode deltas → add to `client_updates[id]['rebuilt_weights']`
3. Call twice with different keys
4. Return tuple

### Phase 3: Add Deltas to Other Algorithms
For comparison, add delta support to:
- fedprox.py
- fedbuff.py  
- fedlama.py

---

## Metrics Tracked

### Current (server.py lines 157-166)
```python
'round': round_num,
'test_accuracy': float(test_accuracy),      # Standard FedAvg/FLIPS
'test_loss': float(test_loss),
'num_selected_clients': ...,
'avg_compression_bytes': ...,
'avg_local_accuracy': ...,
'delta_accuracy': float(delta_accuracy), # From delta coding
'delta_loss': float(delta_loss)
```

### Additional Metrics to Add (Optional)
- `delta_accuracy_vs_standard`: delta_acc - standard_acc
- `compression_ratio`: delta_size / original_size
- `round_overhead`: time for delta decoding

---

## Running the Comparison

### Option A: Run with fedavg only
```bash
python run_experiment.py --rounds 10
# Edit run_experiment.py: algorithms = ['fedavg']
```

### Option B: Run multiple algorithms
```python
algorithms = ['flips', 'fedavg', 'fedbuff', 'fedprox', 'fedlama']
```

### Output Files
- `new_results/metrics_with_delta.json` - All rounds metrics
- Use plotting.py to visualize comparison

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/client.py` | Train local + delta computation |
| `src/server.py` | Aggregate + evaluate both |
| `src/aggregators/fedavg.py` | FedAvg with delta (working) |
| `src/aggregators/flips_aggregator.py` | FLIPS importance-weighted |
| `src/aggregators/delta_decompress.py` | Decode functions |
| `src/plotting.py` | Visualize results |

---

## Next Steps Summary

1. **Verify FedAvg comparison works** - Run and check metrics show both accuracies
2. **Fix FLIPS aggregator** - Implement Option B
3. **Add deltas to fedavg/fedlama client** - So they also produce deltas
4. **Run full comparison** - All algorithms with both paths
5. **Analyze results** - Delta coding accuracy impact