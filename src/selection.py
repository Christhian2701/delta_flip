"""
FLIPS Client Selection logic.
"""
import numpy as np

class ClientSelector:
    """Phase 3: Multi-factor Client Selection."""
    def __init__(self, base_station, config):
        self.base_station = base_station
        self.config = config
        
    def select_clients(self, vehicles, clients_map, k=10):
        """
        Select Top-K clients based on score (Eq 398).
        S_k = b1*A + b2*N + b3*(1/T) + b4*(1-D) + b5*tau
        """
        scores = []
        rssi_threshold = self.config.get('rssi_min', -80)
        grid_size = self.config.get('manhattan_grid_size', 1000)
        density_radius = 200 # Define neighbor radius
        
        # Check algorithm
        algo = self.config.get('algorithm', 'flips')
        is_random_selection = algo in ['fedavg', 'fedbuff', 'fedprox', 'fedlama']
        
        # Betas
        b1 = self.config.get('beta1_accuracy', 0.2)
        b2 = self.config.get('beta2_datasize', 0.2)
        b3 = self.config.get('beta3_time', 0.2)
        b4 = self.config.get('beta4_density', 0.2)
        b5 = self.config.get('beta5_contact', 0.2)
        
        # Pre-calculate global maxes for normalization
        all_samples = [clients_map[v.vehicle_id].num_samples for v in vehicles]
        max_data = max(all_samples) if all_samples else 1
        
        all_times = [clients_map[v.vehicle_id].training_time for v in vehicles]
        max_time = max(all_times) if all_times else 1
        
        # Calculate Density per vehicle
        # D_k = neighbors / total_vehicles
        densities = {}
        for v in vehicles:
            count = 0
            for neighbor in vehicles:
                if v == neighbor: continue
                dist = np.sqrt((v.x - neighbor.x)**2 + (v.y - neighbor.y)**2)
                if dist < density_radius:
                    count += 1
            densities[v.vehicle_id] = count / len(vehicles)

        for v in vehicles:
            if v.vehicle_id not in clients_map: continue
            client = clients_map[v.vehicle_id]
            
            # 1. RSSI Filter (Always apply min connectivity)
            rssi = self.base_station.compute_rssi(v)
            if rssi < rssi_threshold:
                continue
            
            # Update client's localized RSSI
            client.rssi_norm = (rssi - rssi_threshold) / ((-30) - rssi_threshold) # Approx norm
            client.rssi_norm = np.clip(client.rssi_norm, 0, 1)

            # If FedAvg or FedBuff, score is random (but we respect connectivity)
            if is_random_selection:
                score = np.random.random()
                scores.append((client, score, rssi, 0.0))
                continue

            # 2. Compute Score
            # Factors
            cont_time = v.estimate_contact_time(self.base_station)
            norm_cont = min(cont_time / 60.0, 1.0) # Norm against 1 min
            
            norm_data = client.num_samples / max_data
            norm_acc = client.local_accuracy # Already 0-1
            
            # Inverse Training Time (1/T) normalized? 
            # Paper says "normalized inverse training time".
            # Let's use 1 - (time / max_time) as proxy for "fast is good" or similar
            # Or (min_time / time). 
            # If we strictly strictly follow "normalized inverse":
            # inv_t = 1.0 / client.training_time
            # norm_inv_t = inv_t / (1.0/min_time)
            # checking code vs paper text...
            # "beta_3 * T_k^{-1}"
            norm_inv_time = (min(all_times) / client.training_time) if client.training_time > 0 else 0
            
            norm_density = densities[v.vehicle_id]
            
            # Eq 398
            score = b1*norm_acc + b2*norm_data + b3*norm_inv_time + b4*(1.0 - norm_density) + b5*norm_cont
            
            scores.append((client, score, rssi, cont_time))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [x[0] for x in scores[:k]]
        
        return selected
