"""
FLIPS Simulation entry point.
"""
import numpy as np
from mobility import Vehicle, BaseStation
from selection import ClientSelector
from server import FLIPSServer
from client import FLIPSClient

def run_federated_learning(clients, server, test_data, config):
    """
    Run complete federated learning training.
    Phase 3: Integration of Mobility, Network, and Client Selection
    
    Args:
        clients: List of FLIPSClient objects
        server: FLIPSServer object
        test_data: Tuple (X_test, y_test)
        config: Configuration dict

    Returns:
        server: Trained server with metrics
    """
    print(f"\nStarting Federated Learning (FLIPS Phase 3)")
    print(f"Num clients: {len(clients)}")
    print(f"Clients per round: {config['clients_per_round']}")
    print(f"Num rounds: {config['num_rounds']}")
    print("-" * 60)
    
    # Phase 3 Setup: Mobility & Network
    # Create Base Station at center of grid
    grid_size = config.get('manhattan_grid_size', 1000)
    bs_pos = (grid_size/2, grid_size/2)
    base_station = BaseStation(bs_pos, config)
    
    # Create Vehicles (one per client)
    vehicles = []
    client_map = {}
    for client in clients:
        v = Vehicle(client.client_id, config)
        vehicles.append(v)
        client_map[v.vehicle_id] = client
        
    # Create Client Selector
    selector = ClientSelector(base_station, config)

    # Main Loop
    for round_num in range(config['num_rounds']):
        
        # 1. Update Mobility
        # Simulate time passing (e.g. 10 seconds between rounds)
        sim_time_step = 10.0 
        for v in vehicles:
            v.update_position(dt=sim_time_step)
            
            # Update client contact time property
            # Normalize: max contact time ~100s. We use relative factor 
            # or just passed value. Pruning expects 'contact_time' as a factor.
            # estimate_contact_time returns seconds (0-100)
            tau = v.estimate_contact_time(base_station)
            
            # Normalize for pruning/aggregation context (0.0 - 1.0+)
            # If tau is large -> stable connection -> contact_factor > 1?
            # Or just normalize to [0,1] where 1 is "good enough"?
            # Let's normalize by 60s
            norm_tau = min(tau / 60.0, 1.0)
            client = client_map[v.vehicle_id]
            client.contact_time = max(0.1, norm_tau) # Avoid 0
            
        # 2. Select Clients (Phase 3)
        # Filter available vehicles (RSSI > min) and select Top-K
        selected_clients = selector.select_clients(
            vehicles, client_map, k=config['clients_per_round']
        )
        
        # Fallback if too few clients selected (e.g. all out of range)
        if len(selected_clients) < 1:
            print(f"Round {round_num}: No clients selected (all out of range?). Skipping round.")
            continue
            
        # 3. Run Round (Phase 2 logic with Phase 3 selection)
        metrics = server.run_round(round_num, clients, test_data, selected_clients=selected_clients)

        # Print progress
        if round_num % 1 == 0 or round_num == config['num_rounds'] - 1:
            # Get avg contact time of selected
            avg_contact = np.mean([c.contact_time for c in selected_clients])
            
            print(f"Round {round_num:3d} | "
                  f"Acc: {metrics['test_accuracy']:.4f} | "
                  f"Loss: {metrics['test_loss']:.4f} | "
                  f"Comp: {int(metrics['avg_compression_bytes'])} B | "
                  f"Clients: {len(selected_clients)} | "
                  f"AvgTau: {avg_contact:.2f} | "
                  f"delta_acc: {metrics['delta_accuracy']:.4f}")

    print("-" * 60)
    print(f"Final Test Accuracy: {server.round_metrics[-1]['test_accuracy']:.4f}")

    return server
