"""
Main script to run FLIPS experiments
"""

import sys
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import build_cnn, compile_model
from data import load_cifar100_noniid
from client import FLIPSClient
from server import FLIPSServer
from simulation import run_federated_learning


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_metrics(server, output_path='results/metrics.npy'):
    """Save training metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, server.round_metrics)
    print(f"\nMetrics saved to {output_path}")


def plot_results(server, output_path='results/plots'):
    """Plot training results."""
    os.makedirs(output_path, exist_ok=True)

    metrics = server.round_metrics
    rounds = [m['round'] for m in metrics]
    test_acc = [m['test_accuracy'] for m in metrics]
    test_loss = [m['test_loss'] for m in metrics]

    # Accuracy plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, test_acc, 'b-', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title('FLIPS - Test Accuracy vs Rounds')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(rounds, test_loss, 'r-', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Test Loss')
    plt.title('FLIPS - Test Loss vs Rounds')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_curves.png'), dpi=150)
    print(f"Plots saved to {output_path}/training_curves.png")
    plt.close()


def main(args):
    """Main experiment runner with Algorithm Comparison."""
    print("=" * 70)
    print("FLIPS vs FedAvg - Benchmarking Experiment")
    print("=" * 70)

    # Load config
    base_config = load_config(args.config)

    # Override with command line args
    if args.rounds:
        base_config['num_rounds'] = args.rounds
    if args.clients:
        base_config['num_clients'] = args.clients
    if args.epochs:
        base_config['local_epochs'] = args.epochs
    if args.clients_per_round:
        base_config['clients_per_round'] = args.clients_per_round
    if args.mu:
        base_config['mu'] = args.mu

    # Set random seeds
    np.random.seed(base_config['random_seed'])
    import tensorflow as tf
    tf.random.set_seed(base_config['random_seed'])

    # Load data ONCE so partitions are identical for comparison
    print("\nLoading and partitioning data (Shared across algorithms)...")
    client_data, test_data, stats = load_cifar100_noniid(
        num_clients=base_config['num_clients'],
        alpha=base_config['alpha_dirichlet'],
        seed=base_config['random_seed']
    )
    
    # Init Results Storage
    results = {}
    
    # Define algorithms to run
    # Comparison: FLIPS vs FedAvg vs FedBuff vs FedProx vs FedLama
    algorithms = ['flips', 'fedavg', 'fedbuff', 'fedprox', 'fedlama']
    #algorithms = ['flips']
    
    for algo in algorithms:
        print(f"\n" + "#" * 60)
        print(f"RUNNING ALGORITHM: {algo.upper()}")
        print("#" * 60)
        
        # Reset config for this run
        run_config = base_config.copy()
        run_config['algorithm'] = algo
        
        # Re-initialize model (clean slate)
        print(f"Building fresh model for {algo}...")
        model = build_cnn()
        model = compile_model(model, learning_rate=run_config['learning_rate'])
        
        # Re-initialize clients with same data
        clients = []
        for client_id, data in client_data.items():
            client = FLIPSClient(client_id, data, model, run_config)
            clients.append(client)
            
        print(f"Initialized {len(clients)} clients.")
        
        # Re-initialize server
        server = FLIPSServer(model, run_config)
        
        # Run FL
        server = run_federated_learning(clients, server, test_data, run_config)
        
        # Store metrics
        results[algo] = server.round_metrics
        
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    
    # Save results
    results_dir = Path("new_results")
    results_dir.mkdir(exist_ok=True)
    
    out_file = results_dir / "metrics_with_delta.json"
    
    # Convert numpy types to native python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_file, "w") as f:
        # Custom dumper handling
        json.dump(results, f, indent=4, default=convert_numpy)
        print(f"Saved comparative metrics to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FLIPS experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--clients', type=int, default=None,
                       help='Override number of clients')
    # Output arg is less relevant now as we force results/metrics.json for the plotting script
    parser.add_argument('--output', type=str, default='results/metrics.json',
                       help='Path to save metrics (not used in new loop structure)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override local epochs')
    parser.add_argument('--clients_per_round', type=int, default=None,
                       help='Override clients per round')
    parser.add_argument('--mu', type=float, default=0.01,
                       help='Proximal term parameter for FedProx')

    args = parser.parse_args()
    main(args)
