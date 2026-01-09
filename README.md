# FLIPS - Federated Learning with Importance-driven Pruning and Selection

Implementation of the FLIPS algorithm for Vehicular Federated Learning.

## Implementation Status

### ✅ FASE 1: Basic FL + Non-IID Data (COMPLETE)
- [x] 5-layer CNN model (Conv32→Conv64→Conv128→FC256→Softmax)
- [x] CIFAR-100 dataset loading
- [x] Dirichlet non-IID data partitioning (α=0.5)
- [x] FedAvg aggregation
- [x] Basic FL client and server

### ✅ FASE 2: SHAP + Pruning + Quantization (COMPLETE)
- [x] SHAP-based layer importance evaluation (gradient-based approximation)
- [x] Selective layer pruning (Eq 412, 418)
- [x] Model quantization (float32 → float16)
- [x] Importance-weighted aggregation (Eq 440)
- [x] Context factor omega (Eq 305)

### ✅ FASE 3: Network + Mobility + Client Selection (COMPLETE)
- [x] Vehicle mobility simulation (Kalman filter)
- [x] Contact time prediction (Eq 472-476)
- [x] Network simulation (RSSI, bandwidth)
- [x] Multi-factor client selection (Eq 398)
- [x] Complete FLIPS integration (Algorithm 1)

## Quick Start

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run basic FL experiment (FASE 1):
```bash
python run_experiment.py --rounds 20
```

### Run with custom configuration:
```bash
python run_experiment.py --config configs/config.yaml --rounds 50 --clients 30
```

## Project Structure

```
flips_new/
├── src/
│   ├── model.py         # CNN architecture
│   ├── data.py          # CIFAR-100 + Dirichlet partitioning
│   └── flips.py         # FLIPS client, server, and all components
├── configs/
│   └── config.yaml      # Hyperparameters
├── run_experiment.py    # Main experiment script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## SHAP Implementation

This implementation uses **gradient-based layer importance** as an efficient approximation of SHAP values:

### Method
- Computes `importance = mean(|gradient × weight|)` for each layer
- Gradient measures sensitivity of loss to weight changes
- Weight × gradient captures marginal contribution (SHAP concept)
- Uses validation data as specified in Eq 241

### Why Not Full DeepSHAP?
Full coalition-based DeepSHAP (Eq 236-241) is computationally expensive (O(2^L) subsets). Our gradient-based approach:
- ✅ Captures layer contribution to predictions
- ✅ Uses validation data for importance scoring
- ✅ Much faster (enables realistic VFL simulation)
- ✅ Theoretically grounded (gradient = marginal contribution)

### Fallback Methods
1. **Primary**: Gradient × weight importance
2. **Fallback**: Activation variance (if gradients fail)
3. **Last resort**: Weight magnitude

## Configuration

Edit `configs/config.yaml` to change:
- Number of clients, rounds, local epochs
- Dirichlet alpha (data heterogeneity)
- Learning rate, batch size
- SHAP samples (`shap_samples: 100`)
- Pruning parameters (theta_base, alpha_contact)
- Network parameters (RSSI, coverage, bandwidth)

## Results

Results are saved to `results/`:
- `metrics.npy` - Training metrics per round
- `plots/training_curves.png` - Accuracy and loss plots

## Paper Reference

Based on "Federated Learning Framework to Enhance Efficiency and Robustness in Vehicular Scenarios"
See `alg_description` for full paper details.

## Development Timeline

- **Dias 1-2**: FASE 1 - Basic FL ✅
- **Dias 3-4**: FASE 2 - SHAP + Compression
- **Dias 5-7**: FASE 3 - Full FLIPS
