
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_benchmark_results(metrics_file='results/metrics.json'):
    """
    Load metrics.json and plot comparative results for FLIPS vs FedAvg.
    Plots:
    1. Test Accuracy vs Rounds
    2. Test Loss vs Rounds
    3. Avg Data Transmitted vs Rounds
    """
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        results = json.load(f)

    # Setup plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Colors
    colors = {'flips': 'blue', 'fedavg': 'red', 'fedprox': 'green', 'fedlama': 'orange'}
    markers = {'flips': 'o', 'fedavg': 'x', 'fedprox': '^', 'fedlama': 's'}
    
    # 1. Test Accuracy
    ax = axes[0]
    for algo, metrics in results.items():
        rounds = [m['round'] for m in metrics]
        acc = [m['test_accuracy'] for m in metrics]
        label = algo.upper() if algo != 'flips' else 'FLIPS (Ours)'
        ax.plot(rounds, acc, label=label, color=colors.get(algo, 'gray'), 
                marker=markers.get(algo, '.'), linestyle='-', markersize=4)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Accuracy vs Rounds')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Test Loss
    ax = axes[1]
    for algo, metrics in results.items():
        rounds = [m['round'] for m in metrics]
        loss = [m['test_loss'] for m in metrics]
        label = algo.upper() if algo != 'flips' else 'FLIPS (Ours)'
        ax.plot(rounds, loss, label=label, color=colors.get(algo, 'gray'), 
                marker=markers.get(algo, '.'), linestyle='-', markersize=4)

    ax.set_xlabel('Round')
    ax.set_ylabel('Test Loss')
    ax.set_title('Convergence (Loss)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Throughput / Compression
    ax = axes[2]
    for algo, metrics in results.items():
        rounds = [m['round'] for m in metrics]
        # Data in Bytes -> MB
        data_mb = [m['avg_compression_bytes'] / (1024*1024) for m in metrics]
        label = algo.upper() if algo != 'flips' else 'FLIPS (Ours)'
        ax.plot(rounds, data_mb, label=label, color=colors.get(algo, 'gray'), 
                marker=markers.get(algo, '.'), linestyle='--', markersize=4)

    ax.set_xlabel('Round')
    ax.set_ylabel('Avg Update Size (MB)')
    ax.set_title('Communication Overhead')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    
    out_dir = Path(metrics_file).parent
    out_path = out_dir / 'benchmark_plots.png'
    plt.savefig(out_path, dpi=150)
    print(f"Benchmark plots saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_benchmark_results()
