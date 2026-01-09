"""
Data loading and partitioning for FLIPS
Implements CIFAR-100 loading with Dirichlet non-IID partitioning
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_cifar100():
    """
    Load CIFAR-100 dataset.

    Returns:
        (X_train, y_train), (X_test, y_test): Training and test data
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (X_train, y_train), (X_test, y_test)


def partition_data_dirichlet(X, y, num_clients=50, alpha=0.5, num_classes=100, seed=42):
    """
    Partition data using Dirichlet distribution for non-IID split.

    Args:
        X: Feature data (images)
        y: Labels
        num_clients: Number of clients to partition data into
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility

    Returns:
        client_data: Dict {client_id: {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ...}}
    """
    np.random.seed(seed)

    # Get minimum number of samples per class
    min_size = 0
    N = len(y)

    # Organize data by class
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]

    # Sample proportions from Dirichlet distribution
    client_data_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples to clients according to Dirichlet
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        np.random.shuffle(indices)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Distribute indices according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        client_indices_for_class = np.split(indices, proportions)

        # Assign to clients
        for client_id in range(num_clients):
            client_data_indices[client_id].extend(client_indices_for_class[client_id])

    # Create client datasets with train/val split
    client_data = {}
    for client_id in range(num_clients):
        indices = np.array(client_data_indices[client_id])
        np.random.shuffle(indices)

        # 80/20 train/val split
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        client_data[client_id] = {
            'X_train': X[train_indices],
            'y_train': y[train_indices],
            'X_val': X[val_indices],
            'y_val': y[val_indices],
            'num_samples': len(train_indices)
        }

    return client_data


def get_data_statistics(client_data, num_classes=100):
    """
    Compute statistics about data distribution across clients.

    Args:
        client_data: Dict of client datasets
        num_classes: Number of classes

    Returns:
        Dict with statistics
    """
    stats = {
        'num_clients': len(client_data),
        'total_samples': sum([data['num_samples'] for data in client_data.values()]),
        'samples_per_client': [],
        'classes_per_client': [],
        'class_distribution': []
    }

    for client_id, data in client_data.items():
        num_samples = data['num_samples']
        stats['samples_per_client'].append(num_samples)

        # Count unique classes
        unique_classes = len(np.unique(data['y_train']))
        stats['classes_per_client'].append(unique_classes)

        # Class distribution for this client
        class_counts = np.bincount(data['y_train'], minlength=num_classes)
        stats['class_distribution'].append(class_counts)

    stats['avg_samples_per_client'] = np.mean(stats['samples_per_client'])
    stats['std_samples_per_client'] = np.std(stats['samples_per_client'])
    stats['avg_classes_per_client'] = np.mean(stats['classes_per_client'])

    return stats


def load_cifar100_noniid(num_clients=50, alpha=0.5, seed=42):
    """
    Load CIFAR-100 with Dirichlet non-IID partitioning.

    Args:
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed

    Returns:
        client_data: Dict of client datasets
        test_data: Tuple (X_test, y_test)
        stats: Data distribution statistics
    """
    print(f"Loading CIFAR-100 with Dirichlet(α={alpha}) non-IID partitioning...")
    print(f"Number of clients: {num_clients}")

    # Load data
    (X_train, y_train), (X_test, y_test) = load_cifar100()

    # Partition data
    client_data = partition_data_dirichlet(
        X_train, y_train,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed
    )

    # Get statistics
    stats = get_data_statistics(client_data)

    print(f"\nData Statistics:")
    print(f"  Total training samples: {stats['total_samples']}")
    print(f"  Avg samples per client: {stats['avg_samples_per_client']:.1f} ± {stats['std_samples_per_client']:.1f}")
    print(f"  Avg classes per client: {stats['avg_classes_per_client']:.1f}")
    print(f"  Min samples: {min(stats['samples_per_client'])}")
    print(f"  Max samples: {max(stats['samples_per_client'])}")

    return client_data, (X_test, y_test), stats


if __name__ == "__main__":
    # Test data loading and partitioning
    client_data, test_data, stats = load_cifar100_noniid(num_clients=50, alpha=0.5)

    print(f"\nTest set size: {len(test_data[1])}")
    print(f"\nExample client (ID=0):")
    print(f"  Training samples: {len(client_data[0]['y_train'])}")
    print(f"  Validation samples: {len(client_data[0]['y_val'])}")
    print(f"  Unique classes: {len(np.unique(client_data[0]['y_train']))}")
    print(f"  Class distribution (first 10): {np.bincount(client_data[0]['y_train'])[:10]}")
